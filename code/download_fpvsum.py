#!/usr/bin/env python3
"""Download the 49 FPVSum manifest videos from YouTube with yt-dlp.

Video ids map to YouTube ids through the official FPVSum list (code/data/fpvsum_videolist.csv,
copied from github.com/azuxmioy/fpvsum). YouTube often blocks cloud IPs such as Colab; if most
downloads fail with "Sign in to confirm you're not a bot", either pass browser cookies
(--cookies cookies.txt, exported with a "Get cookies.txt" browser extension) or run this
script on your own computer and upload the folder to Google Drive.

  pip install -U yt-dlp
  python code/download_fpvsum.py --out-dir videos/fpvsum
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avsum.config import DATA_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=str(DATA_DIR / "manifest.csv"))
    ap.add_argument("--videolist", default=str(DATA_DIR / "fpvsum_videolist.csv"))
    ap.add_argument("--cookies", default=None)
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--ids", default="", help="comma-separated subset of FPVSum video ids")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yt = {r["video_name"]: r["youtube_id"] for r in csv.DictReader(open(a.videolist, encoding="utf-8"))}
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["dataset"] == "fpvsum"]
    if a.ids:
        want = {x.strip() for x in a.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["video_id"] in want]
    failed = []
    for i, r in enumerate(rows, 1):
        vid = r["video_id"]
        if any(out.glob(f"{vid}.mp4")) or any(out.glob(f"{vid}.mkv")) or any(out.glob(f"{vid}.webm")):
            continue
        yid = yt.get(vid) or vid.split("_", 1)[1]
        url = f"https://www.youtube.com/watch?v={yid}"
        h = a.max_height
        cmd = [sys.executable, "-m", "yt_dlp", "-f", f"bv*[height<={h}][ext=mp4]/bv*[height<={h}]/b[height<={h}]/b",
               "--no-playlist", "--no-part", "--retries", "5", "-o", str(out / f"{vid}.%(ext)s"), url]
        if a.cookies:
            cmd[3:3] = ["--cookies", a.cookies]
        print(f"[{i}/{len(rows)}] {vid} <- {url}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            failed.append(vid)
    have = [r["video_id"] for r in rows if list(out.glob(r["video_id"] + ".*"))]
    print(f"\n{len(have)}/{len(rows)} FPVSum videos present in {out}. Failed: {failed}")


if __name__ == "__main__":
    main()
