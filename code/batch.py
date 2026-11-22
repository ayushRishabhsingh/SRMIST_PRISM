#!/usr/bin/env python3
"""Process every manifest video found under --video-dir (resumable: finished, valid JSONs are skipped).

  python code/batch.py --video-dir /content/videos/fpvsum --dataset fpvsum --out-dir outputs
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avsum.config import DATA_DIR, REPO_DIR  # noqa: E402
from infer import add_common_args, config_from_args, load_validator, write_json  # noqa: E402

VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def shard_rows(rows, shard):
    si, sn = (int(x) for x in shard.split("/"))
    return [r for j, r in enumerate(rows) if j % sn == si] if sn > 1 else rows


def index_videos(root):
    idx = {}
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.name.endswith(".part"):
            idx.setdefault(p.stem, p)
    return idx


def read_manifest(path, dataset="all"):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if dataset != "all":
        rows = [r for r in rows if r["dataset"] == dataset]
    return rows


def already_done(out_path, val, manifest, classes):
    if not out_path.exists():
        return False
    try:
        d = json.loads(out_path.read_text(encoding="utf-8"))
        errs, _ = val.validate_record(d, str(out_path), manifest, classes)
        return not errs
    except Exception:
        return False


def append_report(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_one(summ, val, manifest, classes, vid, path, out_dir, mdur, feat_cache, report):
    t0 = time.time()
    out_path = Path(out_dir) / f"{vid}.json"
    try:
        rec = summ.process(str(path), vid, mdur, feat_cache)
        errs, warns = val.validate_record(rec, str(out_path), manifest, classes)
        write_json(rec, out_path)
        status = "ok" if not errs else "invalid: " + "; ".join(errs[:3])
        hl = sum(h["segment"][1] - h["segment"][0] for h in rec["highlights"])
        append_report(report, {"video_id": vid, "status": status, "seconds": round(time.time() - t0, 1),
                               "label": rec["label"]["name"], "label_score": rec["label"]["score"],
                               "n_actions": len(rec["actions"]), "n_highlights": len(rec["highlights"]),
                               "highlight_pct": round(100 * hl / rec["duration_sec"], 2)})
        return not errs
    except Exception as ex:
        traceback.print_exc()
        append_report(report, {"video_id": vid, "status": f"error: {ex}"[:300], "seconds": round(time.time() - t0, 1),
                               "label": "", "label_score": "", "n_actions": "", "n_highlights": "", "highlight_pct": ""})
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-dir", required=True, nargs="+", help="one or more folders searched recursively")
    ap.add_argument("--shard", default="0/1", help="i/n: process only every n-th remaining video starting at i (split work across Colab accounts)")
    ap.add_argument("--out-dir", default=str(REPO_DIR / "outputs"))
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.csv"))
    ap.add_argument("--dataset", default="all", choices=["all", "epic_kitchens_100", "fpvsum"])
    ap.add_argument("--feat-cache", default=None, help="folder to cache per-video features (re-runs skip decoding)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated subset of video ids")
    ap.add_argument("--report", default=None)
    add_common_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    val = load_validator()
    manifest = val.load_manifest(a.manifest)
    classes = val.load_classes(val.DEF_CLASSES)
    rows = shard_rows(read_manifest(a.manifest, a.dataset), a.shard)
    if a.ids:
        want = {x.strip() for x in a.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["video_id"] in want]
    vids = {}
    for d in a.video_dir:
        if Path(d).exists():
            for k, v in index_videos(d).items():
                vids.setdefault(k, v)
    report = a.report or str(Path(a.out_dir) / "_batch_report.csv")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    todo, missing = [], []
    for r in rows:
        vid = r["video_id"]
        if not a.overwrite and already_done(Path(a.out_dir) / f"{vid}.json", val, manifest, classes):
            continue
        if vid not in vids:
            missing.append(vid)
            continue
        todo.append((vid, vids[vid], float(r["duration_sec"])))
    print(f"[batch] {len(rows)} manifest rows, {len(todo)} to process, {len(missing)} without a video file")
    if missing:
        print(f"[batch] missing videos (first 10): {missing[:10]}")
    if a.limit:
        todo = todo[:a.limit]
    if not todo:
        return
    from avsum.pipeline import Summarizer
    summ = Summarizer(cfg)
    ok = 0
    for i, (vid, path, mdur) in enumerate(todo, 1):
        print(f"\n[batch] ({i}/{len(todo)}) {vid}")
        ok += run_one(summ, val, manifest, classes, vid, path, a.out_dir, mdur, a.feat_cache, report)
    print(f"\n[batch] finished: {ok}/{len(todo)} valid outputs written to {a.out_dir}")


if __name__ == "__main__":
    main()
