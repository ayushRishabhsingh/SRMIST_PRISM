"""End-to-end summarizer: video file -> official Track-04 JSON record."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import video as V
from .actions import ActionModel
from .config import Config
from .encoder import build_encoder
from .highlights import finalize_segments, select_coverage, select_importance
from .signals import class_embeddings, classify_video, importance_curve, l2n, motion_signal
from .vocab import (HIGHLIGHT_GENERIC_POSITIVE, HIGHLIGHT_NEGATIVE, HIGHLIGHT_POSITIVE, KITCHEN_INTERACTION_NEG,
                    KITCHEN_INTERACTION_POS, label_prompt_groups)

W_KITCHEN = {"sem": 0.5, "motion": 0.15, "novelty": 0.2, "quality": 0.15}
W_SPORT = {"sem": 0.5, "motion": 0.25, "novelty": 0.1, "quality": 0.15}


class Summarizer:
    def __init__(self, cfg: Optional[Config] = None, log=print):
        self.cfg = cfg or Config()
        self.log = log if self.cfg.verbose else (lambda *a, **k: None)
        t0 = time.time()
        self.enc = build_encoder(self.cfg.model, self.cfg.device, self.log)
        self.label_names, self.label_emb = class_embeddings(self.enc, label_prompt_groups())
        self.neg = self.enc.encode_texts(HIGHLIGHT_NEGATIVE)
        self.pos = {lab: self.enc.encode_texts(HIGHLIGHT_POSITIVE.get(lab, []) + HIGHLIGHT_GENERIC_POSITIVE)
                    for lab in self.label_names}
        self.k_pos = self.enc.encode_texts(KITCHEN_INTERACTION_POS + HIGHLIGHT_POSITIVE["cooking"])
        self.k_neg = self.enc.encode_texts(KITCHEN_INTERACTION_NEG)
        self.actions = ActionModel(self.enc, self.cfg, self.log)
        self.log(f"[init] ready in {time.time() - t0:.1f}s")

    # -------------------------------------------------------------------------------------------
    def extract(self, path: str) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        embs, thumbs, bright, sharp = [], [], [], []
        size = int(getattr(self.enc, "image_size", 224))
        for frames, _ in V.iter_frame_batches(path, cfg.sample_fps, size, cfg.batch_size,
                                              cfg.hwaccel, cfg.keyframe_mode, self.log):
            embs.append(self.enc.encode_images(frames))
            th, br, sh = V.frame_stats(frames)
            thumbs.append(th); bright.append(br); sharp.append(sh)
        return {"E": np.concatenate(embs).astype(np.float32), "motion": motion_signal(np.concatenate(thumbs)).astype(np.float32),
                "brightness": np.concatenate(bright), "sharp": np.concatenate(sharp)}

    def process(self, path: str, video_id: Optional[str] = None, manifest_duration: Optional[float] = None,
                feat_cache: Optional[str] = None) -> Dict:
        cfg = self.cfg
        t0 = time.time()
        video_id = video_id or Path(path).stem
        info = V.probe(path)
        feats = None
        cache_file = None
        if feat_cache:
            Path(feat_cache).mkdir(parents=True, exist_ok=True)
            cache_file = Path(feat_cache) / f"{video_id}__{cfg.model_slug}__{cfg.sample_fps:g}fps.npz"
            if cache_file.exists():
                z = np.load(cache_file)
                feats = {k: z[k] for k in z.files}
                if "motion" not in feats and "thumbs" in feats:           # cache from an older build
                    feats["motion"] = motion_signal(feats.pop("thumbs").astype(np.float32))
                self.log(f"[cache] loaded features {cache_file.name}")
        if feats is None:
            feats = self.extract(path)
            if cache_file is not None:
                np.savez_compressed(cache_file, E=feats["E"].astype(np.float16), motion=feats["motion"],
                                    brightness=feats["brightness"], sharp=feats["sharp"])
        t_feat = time.time() - t0
        return self.summarize(video_id, feats, info.duration, manifest_duration, t_feat, t0)

    def summarize(self, video_id: str, feats: Dict[str, np.ndarray], probed_duration: float,
                  manifest_duration: Optional[float] = None, t_feat: float = 0.0, t0: Optional[float] = None) -> Dict:
        cfg = self.cfg
        t0 = t0 or time.time()
        fps = cfg.sample_fps
        E = l2n(feats["E"].astype(np.float32))
        T = len(E)
        if T == 0:
            raise RuntimeError(f"{video_id}: no frames")
        duration = probed_duration if probed_duration and probed_duration > 0 else T / fps
        if manifest_duration and abs(duration - manifest_duration) > cfg.duration_tolerance:
            self.log(f"[warn] {video_id}: probed duration {duration:.2f}s vs manifest {manifest_duration:.2f}s -> using manifest")
            duration = float(manifest_duration)
        n_keep = max(1, min(T, int(np.ceil(duration * fps))))
        if n_keep < T:                                   # container longer than the official duration
            E = E[:n_keep]
            feats = {k: (v[:n_keep] if hasattr(v, "__len__") and len(v) == T else v) for k, v in feats.items()}
            T = n_keep
        scale = float(getattr(self.enc, "logit_scale", 100.0))

        label, lscore, probs = classify_video(E, self.label_names, self.label_emb, scale)
        p_kitchen = probs.get("cooking", 0.0)
        kitchen = label == "cooking" or p_kitchen >= cfg.kitchen_threshold
        motion = feats["motion"].astype(np.float32)
        if kitchen:
            imp, sem = importance_curve(E, motion, feats["brightness"], feats["sharp"], self.k_pos, self.k_neg, fps, W_KITCHEN)
            raw = select_coverage(E, imp, fps, duration, cfg)
            actions = self.actions.predict(E, fps, duration, sem)
        else:
            imp, sem = importance_curve(E, motion, feats["brightness"], feats["sharp"], self.pos[label], self.neg, fps, W_SPORT)
            raw = select_importance(imp, fps, duration, cfg)
            actions = []
        highlights = finalize_segments(raw, duration, cfg)
        rec = {
            "team_id": cfg.team_id,
            "video_id": video_id,
            "duration_sec": round(float(duration), 2),
            "label": {"name": label, "score": round(float(min(1.0, max(0.0, lscore))), 4)},
            "actions": actions,
            "highlights": [{"segment": s} for s in highlights],
        }
        tot = sum(e - s for s, e in highlights)
        self.log(f"[done] {video_id}: {duration:.1f}s, label={label} ({lscore:.2f}, p_cooking={p_kitchen:.2f}), "
                 f"mode={'coverage' if kitchen else 'importance'}, {len(highlights)} highlights = {tot:.1f}s "
                 f"({100 * tot / duration:.1f}%), {len(actions)} actions | features {t_feat:.1f}s, total {time.time() - t0:.1f}s")
        return rec
