"""Vision-language encoders.

`VLMEncoder` wraps a Hugging Face CLIP/SigLIP-style dual encoder. To stay robust across
`transformers` versions it calls the full model forward (which always returns
`image_embeds` / `text_embeds`) with a tiny dummy partner input, instead of relying on
`get_*_features` whose return type changed between releases.

`DummyEncoder` is a deterministic stand-in used only for offline tests (no downloads).
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import WEIGHTS_DIR


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class _TextCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, np.ndarray] = {}
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self.data = pickle.load(f)
            except Exception:
                self.data = {}
        self.dirty = False

    def get(self, t):
        return self.data.get(t)

    def put(self, t, v):
        self.data[t] = v.astype(np.float16)
        self.dirty = True

    def save(self):
        if not self.dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(self.data, f)
            os.replace(tmp, self.path)
            self.dirty = False
        except Exception:
            pass


def resolve_model_path(model: str) -> str:
    """Prefer a local copy in weights/hf/<slug> (created by weights/download_weights.py)."""
    local = WEIGHTS_DIR / "hf" / model.replace("/", "__")
    if local.exists() and any(local.iterdir()):
        return str(local)
    return model


class VLMEncoder:
    def __init__(self, model: str, device: str = "auto", log=print):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.name = model
        path = resolve_model_path(model)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        log(f"[encoder] loading {path} on {device} ({self.dtype})")
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=self.dtype).to(device=device, dtype=self.dtype).eval()
        cfg = self.model.config
        self.model_type = str(getattr(cfg, "model_type", "")).lower()
        self.is_siglip = "siglip" in self.model_type
        vc = getattr(cfg, "vision_config", None)
        self.image_size = int(getattr(vc, "image_size", 224) or 224)
        ip = getattr(self.processor, "image_processor", None)
        mean = getattr(ip, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(ip, "image_std", None) or [0.5, 0.5, 0.5]
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        tc = getattr(cfg, "text_config", None)
        self.max_len = 64 if self.is_siglip else int(getattr(tc, "max_position_embeddings", 77) or 77)
        try:
            self.logit_scale = float(self.model.logit_scale.exp().item())
        except Exception:
            self.logit_scale = 100.0
        self.text_cache = _TextCache(WEIGHTS_DIR / "text_cache" / f"{model.replace('/', '__')}.pkl")
        self._dummy_ids = self._tokenize(["a photo"])
        self._dummy_px = torch.zeros(1, 3, self.image_size, self.image_size, dtype=self.dtype, device=device)

    # --- helpers -------------------------------------------------------------------------------
    def _tokenize(self, texts: List[str]):
        tok = getattr(self.processor, "tokenizer", self.processor)
        if self.is_siglip:
            enc = tok(texts, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        else:
            enc = tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        out = {"input_ids": enc["input_ids"].to(self.device)}
        if "attention_mask" in enc and not self.is_siglip:
            out["attention_mask"] = enc["attention_mask"].to(self.device)
        return out

    def _forward(self, ids, px):
        out = self.model(**ids, pixel_values=px)
        if isinstance(out, (tuple, list)):          # very old/odd configs with return_dict=False
            return out[3], out[2]
        return out.image_embeds, out.text_embeds

    # --- public API ------------------------------------------------------------------------------
    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        """frames: uint8 [B,H,W,3] -> L2-normalised float32 [B,D]."""
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
                x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size),
                                                    mode="bilinear", align_corners=False, antialias=True)
            x = ((x - self.mean) / self.std).to(self.dtype)
            img, _ = self._forward(self._dummy_ids, x)
            img = img.float()
            img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return img.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: List[str], batch: int = 256) -> np.ndarray:
        torch = self.torch
        missing = [t for t in dict.fromkeys(texts) if self.text_cache.get(t) is None]
        with torch.inference_mode():
            for i in range(0, len(missing), batch):
                chunk = missing[i:i + batch]
                ids = self._tokenize(chunk)
                _, txt = self._forward(ids, self._dummy_px)
                txt = txt.float()
                txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                for t, v in zip(chunk, txt.cpu().numpy()):
                    self.text_cache.put(t, v)
        if missing:
            self.text_cache.save()
        return _l2n(np.stack([self.text_cache.get(t).astype(np.float32) for t in texts]))


class DummyEncoder:
    """Deterministic, download-free encoder for tests. NOT for real predictions."""

    def __init__(self, dim: int = 64, image_size: int = 64, **_):
        self.name = "dummy"
        self.dim = dim
        self.image_size = image_size
        self.logit_scale = 50.0
        rng = np.random.default_rng(0)
        self.proj = rng.standard_normal((8 * 8 * 3, dim)).astype(np.float32)

    def encode_images(self, frames: np.ndarray) -> np.ndarray:
        b, h, w, _ = frames.shape
        k = h // 8
        small = frames[:, :k * 8, :k * 8].reshape(b, 8, k, 8, k, 3).mean(axis=(2, 4)) / 255.0 - 0.5
        return _l2n(small.reshape(b, -1).astype(np.float32) @ self.proj)

    def encode_texts(self, texts: List[str], batch: int = 256) -> np.ndarray:
        out = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32))
        return _l2n(np.stack(out))


def build_encoder(model: str, device: str = "auto", log=print):
    if model == "dummy":
        return DummyEncoder()
    return VLMEncoder(model, device=device, log=log)
