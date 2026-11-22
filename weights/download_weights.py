#!/usr/bin/env python3
"""Fetch the vision-language model into weights/hf/<model> and pre-compute all text embeddings.

The model (google/siglip-so400m-patch14-384, ~3.5 GB, Apache-2.0) is too large for GitHub, so
it is downloaded from Hugging Face. After this script runs, inference is fully offline:
  python weights/download_weights.py            # default model
  HF_HUB_OFFLINE=1 ./run.sh video.mp4 out.json  # no network needed afterwards
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from avsum.config import DEFAULT_MODEL, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    dst = HERE / "hf" / a.model.replace("/", "__")
    dst.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=a.model, local_dir=str(dst),
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "spiece.model", "*.tiktoken"],
                      ignore_patterns=["*.msgpack", "*.h5", "*.bin", "*.onnx", "*.ot"])
    print(f"model saved to {dst}")
    from avsum.pipeline import Summarizer
    cfg = Config()
    cfg.model = a.model
    Summarizer(cfg)  # encodes and caches every prompt (weights/text_cache/)
    print("text embeddings cached - ready for offline inference")


if __name__ == "__main__":
    main()
