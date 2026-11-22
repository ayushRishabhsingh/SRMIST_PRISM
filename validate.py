#!/usr/bin/env python3
"""Output validator for Track 04 (standard library only).

The student package lists an official `validate.py`, but it was missing from our copy of the
kit, so this file re-implements every rule documented in TEAM_README.md section 3-4. If you
have the official script, run it too - it is the final authority.

  python validate.py --file outputs/P02_109.json -v
  python validate.py --dir outputs/                 # also checks all 187 manifest videos exist
"""
import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_MANIFEST = os.path.join(HERE, "code", "data", "manifest.csv")
DEF_CLASSES = os.path.join(HERE, "code", "data", "supported_classes.json")
REQUIRED = ["team_id", "video_id", "duration_sec", "label", "actions", "highlights"]


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def load_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["video_id"]: (float(r["duration_sec"]), r.get("dataset", "")) for r in csv.DictReader(f)}


def load_classes(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_segment(seg, dur, where, errors):
    if not isinstance(seg, list) or len(seg) != 2 or not all(is_num(v) for v in seg):
        errors.append(f"{where}: segment must be [start, end] numbers, got {seg!r}")
        return None
    s, e = seg
    if not (0 <= s < e):
        errors.append(f"{where}: need 0 <= start < end, got {seg}")
        return None
    if dur is not None and e > dur + 0.5 + 1e-9:
        errors.append(f"{where}: end {e} > duration_sec + 0.5 ({dur + 0.5:.2f})")
        return None
    return s, e


def validate_record(d, fname=None, manifest=None, classes=None):
    errors, warnings = [], []
    if not isinstance(d, dict):
        return ["top level must be a JSON object"], warnings
    for k in REQUIRED:
        if k not in d:
            errors.append(f"missing required field '{k}'")
    if errors:
        return errors, warnings
    extra = set(d) - set(REQUIRED)
    if extra:
        warnings.append(f"extra top-level fields ignored: {sorted(extra)}")
    if not isinstance(d["team_id"], str) or not d["team_id"].strip():
        errors.append("team_id must be a non-empty string")
    vid = d["video_id"]
    if not isinstance(vid, str) or not vid.strip():
        errors.append("video_id must be a non-empty string")
    elif fname is not None and os.path.splitext(os.path.basename(fname))[0] != vid:
        errors.append(f"video_id '{vid}' does not match file name '{os.path.basename(fname)}'")
    elif manifest and vid not in manifest:
        errors.append(f"video_id '{vid}' is not in manifest.csv")
    dur = d["duration_sec"]
    if not is_num(dur) or dur <= 0:
        errors.append(f"duration_sec must be a positive number, got {dur!r}")
        dur = None
    elif manifest and isinstance(vid, str) and vid in manifest:
        md = manifest[vid][0]
        if abs(dur - md) > 0.5 + 1e-9:
            errors.append(f"duration_sec {dur} differs from manifest {md} by more than 0.5 s")
    lab = d["label"]
    if not isinstance(lab, dict) or not isinstance(lab.get("name"), str) or not lab.get("name"):
        errors.append("label must be {\"name\": string, \"score\": number}")
    else:
        if not is_num(lab.get("score")) or not (0 <= lab["score"] <= 1):
            errors.append(f"label.score must be a number in [0, 1], got {lab.get('score')!r}")
        if classes and lab["name"] not in classes.get("labels", []):
            errors.append(f"label.name '{lab['name']}' not in supported labels")
    acts = d["actions"]
    if not isinstance(acts, list):
        errors.append("actions must be an array")
    else:
        if len(acts) > 2000:
            errors.append(f"actions has {len(acts)} items (max 2000)")
        vs = set(classes.get("verbs", [])) if classes else None
        ns = set(classes.get("nouns", [])) if classes else None
        bad_vocab = 0
        for i, a in enumerate(acts):
            w = f"actions[{i}]"
            if not isinstance(a, dict):
                errors.append(f"{w}: must be an object"); continue
            for k in ("segment", "verb", "noun", "score"):
                if k not in a:
                    errors.append(f"{w}: missing '{k}'")
            if "segment" in a:
                check_segment(a["segment"], dur, w, errors)
            if not isinstance(a.get("verb"), str) or not isinstance(a.get("noun"), str):
                errors.append(f"{w}: verb and noun must be strings")
            elif vs is not None and (a["verb"] not in vs or a["noun"] not in ns):
                bad_vocab += 1
            if not is_num(a.get("score")) or not (0 <= a.get("score") <= 1):
                errors.append(f"{w}: score must be a number in [0, 1]")
            if len(errors) > 50:
                errors.append("... too many errors"); break
        if bad_vocab:
            warnings.append(f"{bad_vocab} actions use verbs/nouns outside supported_classes.json")
    hls = d["highlights"]
    if not isinstance(hls, list) or len(hls) < 1:
        errors.append("highlights must be a non-empty array")
    else:
        segs = []
        for i, h in enumerate(hls):
            w = f"highlights[{i}]"
            if not isinstance(h, dict) or "segment" not in h:
                errors.append(f"{w}: must be {{\"segment\": [start, end]}}"); continue
            r = check_segment(h["segment"], dur, w, errors)
            if r:
                segs.append(r)
        if len(segs) == len(hls) and segs:
            for i in range(1, len(segs)):
                if segs[i][0] < segs[i - 1][0]:
                    errors.append(f"highlights not chronologically sorted at index {i}")
                elif segs[i][0] < segs[i - 1][1]:
                    errors.append(f"highlights[{i - 1}] and highlights[{i}] overlap")
                elif segs[i][0] == segs[i - 1][1]:
                    warnings.append(f"highlights[{i - 1}] and highlights[{i}] touch (zero gap)")
            lens = [e - s for s, e in segs]
            if not any(L >= 4.0 for L in lens):
                errors.append("no highlight segment is >= 4.0 s")
            short = sum(1 for L in lens if L < 4.0)
            if short:
                warnings.append(f"{short} highlight segment(s) shorter than 4.0 s (ignored when scoring)")
            if dur:
                tot = sum(lens)
                if tot > 0.15 * dur + 1e-9:
                    warnings.append(f"total highlight length {tot:.2f} s = {100 * tot / dur:.1f}% > 15% budget (soft penalty)")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--manifest", default=DEF_MANIFEST)
    ap.add_argument("--classes", default=DEF_CLASSES)
    ap.add_argument("-v", "--verbose", action="store_true", help="print warnings")
    a = ap.parse_args()
    manifest = load_manifest(a.manifest)
    classes = load_classes(a.classes)
    files = [a.file] if a.file else sorted(os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith(".json"))
    n_err, team_ids = 0, set()
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as ex:
            print(f"[FAIL] {fp}: cannot parse JSON ({ex})"); n_err += 1; continue
        errs, warns = validate_record(d, fp, manifest, classes)
        if isinstance(d, dict) and isinstance(d.get("team_id"), str):
            team_ids.add(d["team_id"])
        if errs:
            n_err += 1
            print(f"[FAIL] {fp}")
            for e in errs:
                print(f"    error: {e}")
        elif a.verbose:
            print(f"[ OK ] {fp}")
        if a.verbose:
            for w in warns:
                print(f"    warning: {w}")
    if len(team_ids) > 1:
        print(f"[FAIL] team_id differs across files: {sorted(team_ids)}"); n_err += 1
    if any(t == "TEAM_ID_NOT_SET" for t in team_ids):
        print("[WARN] team_id is still the placeholder TEAM_ID_NOT_SET - set TEAM_ID before the final run")
    if a.dir and manifest:
        have = {os.path.splitext(os.path.basename(f))[0] for f in files}
        missing = sorted(set(manifest) - have)
        if missing:
            print(f"[WARN] {len(missing)} of {len(manifest)} manifest videos have no JSON yet, e.g. {missing[:5]}")
    print(f"\nChecked {len(files)} file(s): {len(files) - n_err} passed, {n_err} failed.")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
