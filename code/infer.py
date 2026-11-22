#!/usr/bin/env python3
"""Single-video inference (the official entry point is ./run.sh <video.mp4> <out.json>)."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avsum.config import DATA_DIR, REPO_DIR, Config  # noqa: E402


def load_validator():
    import importlib.util
    spec = importlib.util.spec_from_file_location("avsum_validate", REPO_DIR / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def manifest_duration(video_id, manifest_path):
    """Official durations are only used as a sanity check on the probed container duration."""
    if not manifest_path or not os.path.exists(manifest_path):
        return None
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["video_id"] == video_id:
                return float(r["duration_sec"])
    return None


def write_json(rec, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, out_path)


def add_common_args(ap):
    ap.add_argument("--team-id", default=None, help="overrides $TEAM_ID / team_id.txt")
    ap.add_argument("--model", default=None, help="Hugging Face model id or local path ('dummy' for tests)")
    ap.add_argument("--fps", type=float, default=None, help="sampling rate (default 2.0)")
    ap.add_argument("--device", default=None, choices=[None, "auto", "cuda", "cpu"])
    ap.add_argument("--hwaccel", default=None, choices=[None, "auto", "cuda", "none"])
    ap.add_argument("--keyframes", default=None, choices=[None, "auto", "on", "off"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")


def config_from_args(a) -> Config:
    cfg = Config()
    if a.team_id:
        cfg.team_id = a.team_id
    if a.model:
        cfg.model = a.model
    if a.fps:
        cfg.sample_fps = a.fps
    if a.device:
        cfg.device = a.device
    if a.hwaccel:
        cfg.hwaccel = a.hwaccel
    if a.keyframes:
        cfg.keyframe_mode = a.keyframes
    if a.batch_size:
        cfg.batch_size = a.batch_size
    cfg.verbose = not a.quiet
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("out_json")
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.csv"))
    add_common_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    if cfg.team_id == "TEAM_ID_NOT_SET":
        print("[warn] team_id not set: export TEAM_ID=<CollegeName_TeamName> or write it to team_id.txt", file=sys.stderr)
    from avsum.pipeline import Summarizer
    summ = Summarizer(cfg)
    vid = Path(a.video).stem
    rec = summ.process(a.video, vid, manifest_duration(vid, a.manifest))
    val = load_validator()
    errs, warns = val.validate_record(rec, a.out_json if Path(a.out_json).stem == vid else None,
                                      None, val.load_classes(val.DEF_CLASSES))
    for w in warns:
        print(f"[validate] warning: {w}", file=sys.stderr)
    write_json(rec, a.out_json)
    if errs:
        for e in errs:
            print(f"[validate] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
