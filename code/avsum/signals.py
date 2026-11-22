"""Numeric helpers + video-level label classification + per-frame importance."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def robust_z(x: np.ndarray, clip: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    if mad < 1e-9:
        sd = x.std()
        mad = sd if sd > 1e-9 else 1.0
    return np.clip((x - med) / mad, -clip, clip)


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing along axis 0 with edge padding (sigma in samples)."""
    if sigma <= 0 or len(x) < 3:
        return x
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    if x.ndim == 1:
        xp = np.pad(x, (r, r), mode="edge")
        return np.convolve(xp, k, mode="valid")
    xp = np.pad(x, ((r, r),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    out = np.zeros_like(x, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * xp[i:i + len(x)]
    return out


def moving_avg_rows(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average over time (axis 0) for a [T, M] matrix."""
    if n <= 1 or len(x) < 2:
        return x
    c = np.cumsum(np.pad(x, ((1, 0), (0, 0))), axis=0)
    T = len(x)
    half = n // 2
    lo = np.clip(np.arange(T) - half, 0, T)
    hi = np.clip(np.arange(T) - half + n, 0, T)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def rank01(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.ones_like(x, dtype=np.float64)
    r = np.argsort(np.argsort(x))
    return r / (len(x) - 1)


# ---------------------------------------------------------------------------------------------
def class_embeddings(enc, groups: List[Tuple[str, List[str]]]) -> Tuple[List[str], np.ndarray]:
    names, embs = [], []
    for name, prompts in groups:
        t = enc.encode_texts(prompts)
        embs.append(l2n(t.mean(axis=0)))
        names.append(name)
    return names, np.stack(embs)


def classify_video(E: np.ndarray, names: List[str], C: np.ndarray, scale: float,
                   exclude=("other",)) -> Tuple[str, float, Dict[str, float]]:
    """Zero-shot video label: average of per-frame softmax and mean-embedding softmax."""
    sims = E @ C.T
    p_frame = softmax(scale * sims, axis=1).mean(axis=0)
    p_mean = softmax(scale * (l2n(E.mean(axis=0)) @ C.T))
    p = 0.5 * p_frame + 0.5 * p_mean
    mask = np.array([n not in exclude for n in names])
    pk = np.where(mask, p, 0.0)
    pk = pk / pk.sum()
    i = int(np.argmax(pk))
    return names[i], float(pk[i]), {n: float(v) for n, v in zip(names, pk)}


def motion_signal(thumbs: np.ndarray) -> np.ndarray:
    if len(thumbs) < 2:
        return np.zeros(len(thumbs))
    d = np.abs(np.diff(thumbs, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[d[0]], d])


def quality_signal(brightness: np.ndarray, sharp: np.ndarray) -> np.ndarray:
    q = robust_z(np.log1p(sharp))
    q = q - 2.0 * (brightness < 0.06) - 1.0 * (brightness > 0.97)
    return q


def importance_curve(E: np.ndarray, motion: np.ndarray, brightness: np.ndarray, sharp: np.ndarray,
                     pos: np.ndarray, neg: np.ndarray, fps: float, weights: Dict[str, float],
                     smooth_sec: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame importance. Returns (importance, semantic_score)."""
    sem = (E @ pos.T).mean(axis=1) - (E @ neg.T).mean(axis=1)
    mot = np.asarray(motion, dtype=np.float64)
    nov = 1.0 - E @ l2n(E.mean(axis=0))
    qual = quality_signal(brightness, sharp)
    imp = (weights.get("sem", 0.5) * robust_z(sem) + weights.get("motion", 0.2) * robust_z(mot)
           + weights.get("novelty", 0.15) * robust_z(nov) + weights.get("quality", 0.15) * qual)
    imp = gaussian_smooth(imp, smooth_sec * fps)
    return imp, sem
