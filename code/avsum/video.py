"""Video probing and fast frame sampling through an ffmpeg pipe.

Design notes
- EPIC-KITCHENS videos are 1080p @ 60 fps, so full decoding dominates runtime. We
  (1) decode only key-frames when they are dense enough to give the target fps, else
  (2) decode on the GPU (NVDEC, `-hwaccel cuda`) when available, else (3) plain CPU.
- Frames are scaled straight to the encoder's square input size inside ffmpeg (full
  field of view is kept; egocentric hands/objects often sit at the frame edges).
- A reader thread drains the pipe so ffmpeg keeps decoding while the GPU encodes.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class VideoInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    codec: str


def _run(cmd, timeout=120) -> str:
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"command failed ({out.returncode}): {' '.join(cmd)}\n{out.stderr.decode(errors='ignore')[-2000:]}")
    return out.stdout.decode(errors="ignore")


def _parse_rate(r: str) -> float:
    try:
        if "/" in r:
            a, b = r.split("/")
            return float(a) / float(b) if float(b) else 0.0
        return float(r)
    except Exception:
        return 0.0


def probe(path: str) -> VideoInfo:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found - install ffmpeg (apt-get install -y ffmpeg)")
    txt = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,width,height,codec_name,duration:format=duration",
        "-of", "json", path,
    ])
    j = json.loads(txt)
    st = (j.get("streams") or [{}])[0]
    dur = 0.0
    for cand in (j.get("format", {}).get("duration"), st.get("duration")):
        try:
            if cand is not None and float(cand) > 0:
                dur = float(cand)
                break
        except Exception:
            pass
    fps = _parse_rate(st.get("avg_frame_rate", "0/0")) or _parse_rate(st.get("r_frame_rate", "0/0"))
    return VideoInfo(path=path, duration=dur, fps=fps or 30.0, width=int(st.get("width", 0) or 0),
                     height=int(st.get("height", 0) or 0), codec=str(st.get("codec_name", "")))


def keyframe_interval(path: str, probe_seconds: float = 40.0) -> float:
    """Median spacing (s) between key-frames over the first `probe_seconds`. Uses packets only (no decode)."""
    try:
        txt = _run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"%+{probe_seconds}",
            "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", path,
        ], timeout=60)
    except Exception:
        return float("inf")
    ts = []
    for line in txt.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2 and "K" in parts[1]:
            try:
                ts.append(float(parts[0]))
            except ValueError:
                pass
    ts = sorted(ts)
    if len(ts) < 3:
        return float("inf")
    return float(np.median(np.diff(ts)))


_CUDA_OK: Optional[bool] = None


def cuda_decode_available() -> bool:
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    ok = False
    try:
        if shutil.which("nvidia-smi"):
            txt = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=20).stdout.decode()
            ok = "cuda" in txt
    except Exception:
        ok = False
    _CUDA_OK = ok
    return ok


def _ffmpeg_cmd(path: str, fps: float, size: int, hwaccel: bool, keyframes: bool):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    if keyframes:
        cmd += ["-skip_frame", "nokey"]
    cmd += ["-i", path, "-an", "-sn", "-dn",
            "-vf", f"fps={fps:.6f},scale={size}:{size}:flags=area",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    return cmd


def choose_decode_mode(path: str, fps: float, hwaccel: str = "auto", keyframe_mode: str = "auto") -> Tuple[bool, bool]:
    """Returns (use_cuda, keyframes_only)."""
    keyframes = False
    if keyframe_mode == "on":
        keyframes = True
    elif keyframe_mode == "auto":
        ki = keyframe_interval(path)
        keyframes = ki <= (1.0 / fps) * 1.05
    use_cuda = False
    if not keyframes:
        if hwaccel == "cuda":
            use_cuda = True
        elif hwaccel == "auto":
            use_cuda = cuda_decode_available()
    return use_cuda, keyframes


def iter_frame_batches(path: str, fps: float, size: int, batch: int = 32,
                       hwaccel: str = "auto", keyframe_mode: str = "auto",
                       log=print) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yields (frames uint8 [B,size,size,3], timestamps float [B]) at a constant `fps`.

    Falls back to CPU decoding automatically if GPU decoding fails before producing frames.
    """
    use_cuda, keyframes = choose_decode_mode(path, fps, hwaccel, keyframe_mode)
    attempts = [(use_cuda, keyframes)]
    if use_cuda:
        attempts.append((False, keyframes))
    if keyframes:
        attempts.append((False, False))

    frame_bytes = size * size * 3
    for i, (cu, kf) in enumerate(attempts):
        log(f"[video] decode mode: {'cuda' if cu else 'cpu'}{' keyframes-only' if kf else ''} @ {fps} fps, {size}px")
        cmd = _ffmpeg_cmd(path, fps, size, cu, kf)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)
        q: "queue.Queue" = queue.Queue(maxsize=8)
        err_chunks = []

        def _drain_err():
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                err_chunks.append(chunk)

        def _reader():
            buf = []
            try:
                while True:
                    data = proc.stdout.read(frame_bytes)
                    if not data or len(data) < frame_bytes:
                        break
                    buf.append(np.frombuffer(data, dtype=np.uint8).reshape(size, size, 3))
                    if len(buf) == batch:
                        q.put(np.stack(buf))
                        buf = []
                if buf:
                    q.put(np.stack(buf))
            finally:
                q.put(None)

        te = threading.Thread(target=_drain_err, daemon=True)
        tr = threading.Thread(target=_reader, daemon=True)
        te.start(); tr.start()
        n = 0
        while True:
            item = q.get()
            if item is None:
                break
            ts = (np.arange(n, n + len(item), dtype=np.float64)) / fps
            n += len(item)
            yield item, ts
        tr.join(); proc.wait(); te.join(timeout=5)
        if n > 0:
            if proc.returncode not in (0, None):
                log(f"[video] warning: ffmpeg exited with {proc.returncode} after {n} frames")
            return
        log(f"[video] decode attempt {i + 1} produced no frames: {b''.join(err_chunks).decode(errors='ignore')[-500:]}")
    raise RuntimeError(f"could not decode any frames from {path}")


def frame_stats(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheap per-frame signals from small grayscale thumbnails.

    Returns (thumbs [B,32,32] float32, brightness [B], sharpness [B]).
    """
    f = frames.astype(np.float32)
    gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    b, h, w = gray.shape
    k = max(1, h // 32)
    hh, ww = (h // k) * k, (w // k) * k
    thumbs = gray[:, :hh, :ww].reshape(b, hh // k, k, ww // k, k).mean(axis=(2, 4))
    thumbs = thumbs[:, :32, :32] / 255.0
    brightness = gray.mean(axis=(1, 2)) / 255.0
    # Laplacian variance on a 2x-downsampled image as a blur measure
    g2 = gray[:, ::2, ::2]
    lap = (-4 * g2[:, 1:-1, 1:-1] + g2[:, :-2, 1:-1] + g2[:, 2:, 1:-1] + g2[:, 1:-1, :-2] + g2[:, 1:-1, 2:])
    sharp = lap.reshape(b, -1).var(axis=1)
    return thumbs.astype(np.float32), brightness.astype(np.float32), sharp.astype(np.float32)
