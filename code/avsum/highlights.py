"""Highlight selection under the official constraints.

Rules (TEAM_README): chronologically sorted, strictly non-overlapping segments; each segment
>= 4.0 s (shorter ones are ignored by the scorer); total <= 15 % of the duration.

Two selection modes:
- coverage (kitchen / EPIC): many short (~4.3 s) segments chosen by greedy facility-location
  over frame embeddings, weighted by hand-object-interaction importance. This maximises the
  number of *distinct* actions inside the summary (Action Coverage, 40 %).
- importance (sports / FPVSum): segments of 4-12 s with the highest mean importance, greedily
  packed into the budget and merged when adjacent. This targets the top-15 % human consensus
  frames (Highlight F1, 25 %).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .signals import l2n, rank01

Seg = Tuple[float, float, float]  # (start, end, priority)


def budget_seconds(duration: float, ratio: float, margin: float) -> float:
    return max(0.0, ratio * duration - margin)


def _window_means(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[n:] - c[:-n]) / n


def select_coverage(E: np.ndarray, imp: np.ndarray, fps: float, duration: float, cfg) -> List[Seg]:
    dt = 1.0 / fps
    T = len(E)
    B = budget_seconds(duration, cfg.budget_ratio, cfg.budget_margin_sec)
    minL = cfg.min_seg_sec
    K = max(1, int(B // minL))                     # as many >= 4.1 s segments as the budget allows
    L = max(minL, min(max(cfg.coverage_seg_sec, minL), B / K)) if B >= minL else minL
    if B / K > cfg.coverage_seg_sec:               # long videos: keep segments short, add more of them
        K = max(1, int(B // cfg.coverage_seg_sec))
        L = max(minL, B / K)
    n = max(1, int(round(L / dt)))
    if T <= n or duration <= L + 1e-6:
        return [(0.0, min(duration, L), 1.0)]
    last = int(np.floor((duration - L) / dt))       # last start frame that keeps the segment inside
    stride = max(1, (min(T - n, last) + 1) // 2500)
    starts = np.arange(0, min(T - n, last) + 1, stride)
    cs = np.concatenate([np.zeros((1, E.shape[1]), dtype=np.float64), np.cumsum(E, axis=0, dtype=np.float64)])
    W = l2n((cs[starts + n] - cs[starts]).astype(np.float32))          # candidate window embeddings
    S = (E @ W.T).astype(np.float32)                                     # [T, C]
    wimp = rank01(_window_means(imp, n)[starts])                          # [C] in [0,1]
    cur = np.full(T, float(S.min()), dtype=np.float32)
    blocked = np.zeros(len(starts), dtype=bool)
    beta = 0.6
    chosen: List[Seg] = []
    base = None
    tstart = starts * dt
    for _ in range(K):
        gain = np.maximum(S - cur[:, None], 0.0).sum(axis=0)
        if base is None:
            base = float(gain.mean()) + 1e-6
        obj = (gain + 1e-3 * base) * (1.0 + beta * wimp)
        obj[blocked] = -np.inf
        c = int(np.argmax(obj))
        if not np.isfinite(obj[c]):
            break
        s0 = float(tstart[c])
        chosen.append((s0, min(duration, s0 + L), float(obj[c])))
        cur = np.maximum(cur, S[:, c])
        blocked |= (tstart > s0 - L - cfg.seg_gap_sec) & (tstart < s0 + L + cfg.seg_gap_sec)
    return chosen


def select_importance(imp: np.ndarray, fps: float, duration: float, cfg) -> List[Seg]:
    dt = 1.0 / fps
    T = len(imp)
    B = budget_seconds(duration, cfg.budget_ratio, cfg.budget_margin_sec)
    cands = []
    for L in cfg.f1_seg_lengths:
        n = max(1, int(round(L / dt)))
        if n > T:
            continue
        m = _window_means(imp, n)
        step = max(1, n // 4)
        for a in range(0, T - n + 1, step):
            if a * dt + L <= duration + 1e-6:
                cands.append((float(m[a]), a, n, L))
    if not cands:
        return [(0.0, min(duration, max(cfg.min_seg_sec, T * dt)), 1.0)]
    cands.sort(key=lambda x: -x[0])
    chosen: List[Seg] = []
    total = 0.0
    for score, a, n, L in cands:
        s, e = a * dt, min(duration, a * dt + L)
        if e - s < cfg.min_seg_sec - 1e-6:
            continue
        if chosen and total + (e - s) > B:
            continue
        if any(not (e <= cs - 1e-9 or s >= ce + 1e-9) for cs, ce, _ in chosen):
            continue
        chosen.append((s, e, score))
        total += e - s
        if total >= B - cfg.min_seg_sec:
            # keep scanning only for segments that still fit
            pass
    # merge near-adjacent segments if the gap still fits the budget
    chosen.sort()
    merged: List[Seg] = []
    for s, e, p in chosen:
        if merged:
            ps, pe, pp = merged[-1]
            gap = s - pe
            if 0 <= gap < 1.0 and total + gap <= B:
                merged[-1] = (ps, e, max(pp, p))
                total += gap
                continue
        merged.append((s, e, p))
    return merged


def finalize_segments(segs: List[Seg], duration: float, cfg) -> List[List[float]]:
    """Clip, enforce min length, sort, de-overlap, round to 2 decimals, enforce budget.

    Guarantees: sorted, strictly non-overlapping, all inside [0, duration], at least one
    segment >= 4.0 s, and total <= budget_ratio * duration whenever more than one 4-s segment fits.
    """
    D = math.floor(duration * 100) / 100.0
    minL = min(cfg.min_seg_sec, D) if D > 0 else cfg.min_seg_sec
    # 1) clip + min length
    fixed: List[Seg] = []
    for s, e, p in segs:
        s, e = max(0.0, s), min(D, e)
        if e - s < minL:
            c = 0.5 * (s + e)
            s, e = c - minL / 2, c + minL / 2
            if s < 0:
                s, e = 0.0, minL
            if e > D:
                s, e = max(0.0, D - minL), D
        if e > s:
            fixed.append((s, e, p))
    if not fixed:
        fixed = [(0.0, minL, 1.0)]
    fixed.sort()
    # 2) union overlapping / too-close segments
    out: List[Seg] = []
    for s, e, p in fixed:
        if out and s < out[-1][1] + cfg.seg_gap_sec:
            ps, pe, pp = out[-1]
            out[-1] = (ps, max(pe, e), max(pp, p))
        else:
            out.append((s, e, p))
    # 3) round on a 0.01 s grid (start down, end up, never past D)
    rnd: List[Seg] = []
    for s, e, p in out:
        rs = math.floor(s * 100 + 1e-6) / 100.0
        re_ = min(D, math.ceil(e * 100 - 1e-6) / 100.0)
        if rnd and rs <= rnd[-1][1]:
            rs = rnd[-1][1] + 0.01
        if re_ - rs >= 0.01:
            rnd.append((round(rs, 2), round(re_, 2), p))
    # 4) budget: trim lowest-priority segments first, drop only if trimming is impossible
    hard = math.floor(cfg.budget_ratio * duration * 100) / 100.0
    def total():
        return sum(e - s for s, e, _ in rnd)
    guard = 0
    while len(rnd) > 1 and total() > hard + 1e-9 and guard < 1000:
        guard += 1
        over = total() - hard
        order = sorted(range(len(rnd)), key=lambda i: rnd[i][2])
        done = False
        for i in order:
            s, e, p = rnd[i]
            if (e - s) - over >= minL:
                cut = math.ceil(over * 100 - 1e-6) / 100.0
                c = (s + e) / 2
                ns = round(math.floor((c - ((e - s) - cut) / 2) * 100) / 100.0, 2)
                ns = max(s, ns)
                ne = round(ns + (e - s) - cut, 2)
                rnd[i] = (ns, ne, p)
                done = True
                break
        if not done:
            rnd.pop(order[0])
    if len(rnd) == 1 and total() > hard + 1e-9:
        # short video: one segment unavoidably exceeds 15 %; keep it as short as allowed
        s, e, p = rnd[0]
        L = max(minL, hard)
        if e - s > L:
            c = (s + e) / 2
            ns = max(0.0, math.floor((c - L / 2) * 100) / 100.0)
            ne = min(D, math.ceil((ns + L) * 100 - 1e-6) / 100.0)
            rnd[0] = (round(ns, 2), round(ne, 2), p)
    res = [[round(s, 2), round(e, 2)] for s, e, _ in rnd]
    # 5) guarantee one segment of >= 4.0 s
    if not any(e - s >= 4.0 - 1e-9 for s, e in res):
        if res:
            j = max(range(len(res)), key=lambda i: res[i][1] - res[i][0])
            s, e = res[j]
        else:
            j, s, e = -1, 0.0, minL
        c = 0.5 * (s + e)
        L = max(cfg.min_seg_sec, 4.01)
        ns = max(0.0, min(c - L / 2, D - L))
        ne = min(D, ns + L)
        seg = [round(math.floor(ns * 100) / 100.0, 2), round(math.ceil(ne * 100 - 1e-6) / 100.0, 2)]
        seg[1] = min(seg[1], D)
        res = [seg]  # a lone valid segment beats an invalid file
    return res
