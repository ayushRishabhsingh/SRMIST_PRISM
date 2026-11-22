"""Zero-shot verb-noun action detection for kitchen videos.

Per frame we score three prompt families with the vision-language model:
  verbs ("hands cutting something"), nouns ("a chopping board in a kitchen") and
  verb-noun pairs ("hands cutting the onion"), restricted to the pairs that actually occur in
  the EPIC-KITCHENS-100 *train* split, combined with that split's pair frequency prior:

  log s_t(v,n) = a*log p_pair + b*(log p_verb + log p_noun) + w*log prior(v,n)

Scores are softmax-normalised per frame, smoothed, averaged over multi-scale sliding windows
(1-6 s; EPIC actions have a median length of ~1.6 s), the top-k pairs per window are kept,
class-wise temporal NMS removes duplicates and the best `max_actions` (<= 2000) are returned.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .signals import log_softmax, moving_avg_rows, rank01, softmax
from .vocab import load_classes, load_pair_prior, noun_prompts, pair_prompts, verb_prompts


def _iou(s, e, S, E):
    inter = np.clip(np.minimum(e, E) - np.maximum(s, S), 0, None)
    union = (e - s) + (E - S) - inter
    return inter / np.maximum(union, 1e-9)


class ActionModel:
    def __init__(self, enc, cfg, log=print):
        self.cfg = cfg
        self.enc = enc
        cls = load_classes()
        self.verbs = [v for v in cls["verbs"] if v != "other"]
        self.nouns = [n for n in cls["nouns"] if n != "other"]
        self.pairs, counts = load_pair_prior(cfg.prior_min_count)
        vi = {v: i for i, v in enumerate(self.verbs)}
        ni = {n: i for i, n in enumerate(self.nouns)}
        self.pv = np.array([vi[v] for v, _ in self.pairs])
        self.pn = np.array([ni[n] for _, n in self.pairs])
        c = np.asarray(counts, dtype=np.float64)
        self.log_prior = np.log(c / c.sum())
        log(f"[actions] encoding {len(self.verbs)} verbs, {len(self.nouns)} nouns, {len(self.pairs)} verb-noun pairs")
        self.Tv = enc.encode_texts(verb_prompts(self.verbs))
        self.Tn = enc.encode_texts(noun_prompts(self.nouns))
        self.Tp = enc.encode_texts(pair_prompts(self.pairs))

    def frame_pair_probs(self, E: np.ndarray) -> np.ndarray:
        s = float(getattr(self.enc, "logit_scale", 100.0))
        lv = log_softmax(s * (E @ self.Tv.T), axis=1)
        ln = log_softmax(s * (E @ self.Tn.T), axis=1)
        lp = log_softmax(s * (E @ self.Tp.T), axis=1)
        comb = 0.5 * lp + 0.25 * (lv[:, self.pv] + ln[:, self.pn]) + self.cfg.prior_weight * self.log_prior[None, :]
        return softmax(comb, axis=1).astype(np.float32)

    def predict(self, E: np.ndarray, fps: float, duration: float, actionness: np.ndarray) -> List[Dict]:
        cfg = self.cfg
        dt = 1.0 / fps
        T = len(E)
        if T == 0:
            return []
        q = self.frame_pair_probs(E)
        q = moving_avg_rows(q, max(1, int(round(1.0 / dt)) | 1))
        act = rank01(actionness) if len(actionness) == T else np.ones(T)
        cq = np.concatenate([np.zeros((1, q.shape[1]), dtype=np.float64), np.cumsum(q, axis=0, dtype=np.float64)])
        ca = np.concatenate([[0.0], np.cumsum(act)])
        S, Ee, P, Sc = [], [], [], []
        k = max(1, cfg.action_topk)
        for L in cfg.action_window_secs:
            n = max(1, int(round(L / dt)))
            if n > T:
                continue
            stride = max(1, n // 2)
            a = np.arange(0, T - n + 1, stride)
            m = (cq[a + n] - cq[a]) / n                      # [A, M]
            am = (ca[a + n] - ca[a]) / n                     # [A]
            top = np.argpartition(-m, kth=min(k, m.shape[1] - 1), axis=1)[:, :k]
            sc = np.take_along_axis(m, top, axis=1) * (0.5 + 0.5 * am[:, None])
            S.append(np.repeat(a * dt, k)); Ee.append(np.repeat(np.minimum((a + n) * dt, duration), k))
            P.append(top.reshape(-1)); Sc.append(sc.reshape(-1))
        if not S:
            n = T
            m = q.mean(axis=0)
            top = np.argsort(-m)[:k]
            S, Ee, P, Sc = [np.zeros(k)], [np.full(k, min(duration, T * dt))], [top], [m[top]]
        s = np.concatenate(S); e = np.concatenate(Ee); p = np.concatenate(P); sc = np.concatenate(Sc)
        keep = e > s + 1e-6
        s, e, p, sc = s[keep], e[keep], p[keep], sc[keep]
        # class-wise temporal NMS (scores stay absolute, i.e. comparable across videos for mAP)
        order = np.argsort(-sc)
        s, e, p, sc = s[order], e[order], p[order], sc[order]
        kept_idx = []
        by_cls: Dict[int, List[int]] = {}
        for i in range(len(sc)):
            lst = by_cls.setdefault(int(p[i]), [])
            if lst:
                j = np.array(lst)
                if np.any(_iou(s[i], e[i], s[j], e[j]) > cfg.action_nms_iou):
                    continue
            lst.append(i)
            kept_idx.append(i)
            if len(kept_idx) >= cfg.max_actions:
                break
        out = []
        for i in kept_idx:
            v, nn = self.pairs[int(p[i])]
            st = math.floor(float(s[i]) * 100) / 100.0
            en = min(math.floor(duration * 100) / 100.0, math.ceil(float(e[i]) * 100) / 100.0)
            if en <= st:
                continue
            out.append({"segment": [round(st, 2), round(en, 2)], "verb": v, "noun": nn,
                        "score": round(float(np.clip(sc[i], 0.0, 1.0)) ** 0.5, 6)})
        out.sort(key=lambda x: (x["segment"][0], -x["score"]))
        return out
