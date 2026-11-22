"""Build weights/epic_action_prior.json from the EPIC-KITCHENS-100 TRAIN annotations.

Only the public *train* split is used (never validation): it gives which verb-noun pairs
exist and how frequent they are, plus typical action durations. The evaluation videos are
the validation split, whose annotations are not read anywhere in this repository.

Usage:
  git clone --depth 1 https://github.com/epic-kitchens/epic-kitchens-100-annotations.git
  python code/build_prior.py --ann epic-kitchens-100-annotations
"""
import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avsum.config import WEIGHTS_DIR  # noqa: E402
from avsum.vocab import load_classes  # noqa: E402


def t2s(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="path to epic-kitchens-100-annotations")
    ap.add_argument("--out", default=str(WEIGHTS_DIR / "epic_action_prior.json"))
    a = ap.parse_args()
    ann = Path(a.ann)
    verb_key = {r["id"]: r["key"] for r in csv.DictReader(open(ann / "EPIC_100_verb_classes.csv", encoding="utf-8"))}
    noun_key = {r["id"]: r["key"] for r in csv.DictReader(open(ann / "EPIC_100_noun_classes.csv", encoding="utf-8"))}
    cls = load_classes()
    vs, ns = set(cls["verbs"]), set(cls["nouns"])
    counts = collections.Counter()
    durs = collections.defaultdict(list)
    for r in csv.DictReader(open(ann / "EPIC_100_train.csv", encoding="utf-8")):
        v, n = verb_key[r["verb_class"]], noun_key[r["noun_class"]]
        if v not in vs or n not in ns:
            continue
        counts[(v, n)] += 1
        durs[(v, n)].append(max(0.0, t2s(r["stop_timestamp"]) - t2s(r["start_timestamp"])))
    pairs = [{"verb": v, "noun": n, "count": c, "median_dur": round(statistics.median(durs[(v, n)]), 2)}
             for (v, n), c in counts.most_common()]
    out = {"source": "EPIC-KITCHENS-100 train split (EPIC_100_train.csv) - no validation data used",
           "n_instances": sum(counts.values()), "pairs": pairs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(f"wrote {a.out}: {len(pairs)} pairs from {out['n_instances']} train instances")


if __name__ == "__main__":
    main()
