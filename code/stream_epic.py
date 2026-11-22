#!/usr/bin/env python3
"""Download -> process -> delete pipeline for the 138 EPIC-KITCHENS-100 videos (~184 GB).

Colab's disk cannot hold the whole split, so download workers fetch a few videos ahead
(official resumable downloader, MD5-verified) while the GPU processes the current one; each
video is deleted as soon as its JSON is written. Safe to stop and re-run: finished videos are
skipped and partial downloads resume.

  git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
  python code/stream_epic.py --downloader-dir epic-kitchens-download-scripts \
      --dl-dir /content/epic_tmp --out-dir /content/drive/MyDrive/avsum/outputs
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avsum.config import DATA_DIR  # noqa: E402
from batch import already_done, read_manifest, run_one, shard_rows  # noqa: E402
from infer import add_common_args, config_from_args, load_validator  # noqa: E402


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def find_video(dl_dir, vid):
    for ext in ("MP4", "mp4"):
        hits = list(Path(dl_dir).rglob(f"{vid}.{ext}"))
        if hits:
            return hits[0]
    return None


def download(downloader_dir, dl_dir, vid, retries, log):
    script = "epic_video_downloader_verify.py"
    if not (Path(downloader_dir) / script).exists():
        script = "epic_downloader.py"
        cmd = [sys.executable, script, "--videos", "--specific-videos", vid, "--output-path", str(dl_dir)]
    else:
        cmd = [sys.executable, script, "--specific-videos", vid, "--output-path", str(dl_dir),
               "--max-retries", str(retries), "--no-md5-progress"]
    for attempt in range(1, retries + 1):
        t0 = time.time()
        r = subprocess.run(cmd, cwd=downloader_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p = find_video(dl_dir, vid)
        if p is not None and p.stat().st_size > 0 and r.returncode == 0:
            mb = p.stat().st_size / 1e6
            log(f"[dl] {vid}: {mb:.0f} MB in {time.time() - t0:.0f}s ({mb / max(1, time.time() - t0):.1f} MB/s)")
            return p
        log(f"[dl] {vid}: attempt {attempt} failed (rc={r.returncode}) {r.stdout.decode(errors='ignore')[-400:]}")
        time.sleep(5 * attempt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--downloader-dir", required=True)
    ap.add_argument("--dl-dir", default="/content/epic_tmp")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.csv"))
    ap.add_argument("--workers", type=int, default=3, help="parallel downloads")
    ap.add_argument("--max-ahead", type=int, default=4, help="max downloaded-but-unprocessed videos")
    ap.add_argument("--min-free-gb", type=float, default=15.0)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--keep-videos", action="store_true")
    ap.add_argument("--feat-cache", default=None)
    ap.add_argument("--ids", default="", help="comma-separated subset of video ids")
    ap.add_argument("--order", default="short-first", choices=["short-first", "manifest"])
    ap.add_argument("--shard", default="0/1", help="i/n: this session takes every n-th video (run 0/2 and 1/2 on two accounts)")
    add_common_args(ap)
    a = ap.parse_args()
    cfg = config_from_args(a)
    log = print
    val = load_validator()
    manifest = val.load_manifest(a.manifest)
    classes = val.load_classes(val.DEF_CLASSES)
    rows = read_manifest(a.manifest, "epic_kitchens_100")
    if a.ids:
        want = set(x.strip() for x in a.ids.split(",") if x.strip())
        rows = [r for r in rows if r["video_id"] in want]
    if a.order == "short-first":
        rows.sort(key=lambda r: float(r["duration_sec"]))
    rows = shard_rows(rows, a.shard)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    Path(a.dl_dir).mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not already_done(Path(a.out_dir) / f"{r['video_id']}.json", val, manifest, classes)]
    log(f"[stream] {len(rows)} EPIC videos, {len(todo)} remaining")
    if not todo:
        return
    from avsum.pipeline import Summarizer
    summ = Summarizer(cfg)
    report = str(Path(a.out_dir) / "_epic_report.csv")

    jobs: "queue.Queue" = queue.Queue()
    for r in todo:
        jobs.put(r)
    ready: "queue.Queue" = queue.Queue()
    ahead = threading.Semaphore(a.max_ahead)
    failed = []

    def worker():
        while True:
            try:
                r = jobs.get_nowait()
            except queue.Empty:
                return
            ahead.acquire()
            while free_gb(a.dl_dir) < a.min_free_gb:
                time.sleep(10)
            p = download(a.downloader_dir, a.dl_dir, r["video_id"], a.retries, log)
            if p is None:
                failed.append(r["video_id"])
                ahead.release()
                ready.put((r, None))
            else:
                ready.put((r, p))

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, a.workers))]
    for t in threads:
        t.start()
    done = ok = 0
    t_start = time.time()
    while done < len(todo):
        r, p = ready.get()
        done += 1
        if p is None:
            continue
        vid = r["video_id"]
        log(f"\n[stream] ({done}/{len(todo)}) processing {vid} ({float(r['duration_sec']):.0f}s)")
        ok += run_one(summ, val, manifest, classes, vid, p, a.out_dir, float(r["duration_sec"]), a.feat_cache, report)
        if not a.keep_videos:
            try:
                os.remove(p)
            except OSError:
                pass
        ahead.release()
        el = time.time() - t_start
        log(f"[stream] progress {done}/{len(todo)}, {ok} ok, {len(failed)} download failures, "
            f"{el / 60:.1f} min elapsed, free disk {free_gb(a.dl_dir):.0f} GB")
    log(f"\n[stream] finished: {ok} valid outputs; download failures: {failed}")


if __name__ == "__main__":
    main()
