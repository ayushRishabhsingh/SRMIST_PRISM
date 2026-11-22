# Action-Aware Egocentric Video Summarization — Track 04

Samsung PRISM XR Hackathon 2026 · Theme 04 · Team: `TEAM_ID` (set in `team_id.txt` or the `TEAM_ID` env var)

Given a 2–10 min (up to 33 min) first-person video, the system predicts a **video label**, **verb–noun actions
with time segments**, and a **≈15 % highlight summary** made of segments of the original footage. It writes one JSON
per video in the official 6-field schema.

```
./run.sh <video.mp4> <out.json>
```

---

## 1. How it works

```
 MP4 ──ffmpeg (NVDEC / key-frame decode)──► frames @ 2 fps, 384 px
        │
        ▼
 SigLIP-so400m image encoder ──► frame embeddings E[t]  (+ motion, sharpness, brightness from thumbnails)
        │
        ├─► Video label ......... zero-shot vs. 16 label prompt ensembles (frame-vote + mean-embedding)
        │
        ├─► Kitchen branch (label = cooking)
        │      Actions ......... per-frame verb / noun / verb-noun prompt scores × EPIC-100 train-split pair prior
        │                        → multi-scale windows (1–6 s) → top-3 pairs/window → class-wise NMS → ≤ 2000
        │      Highlights ...... greedy facility-location over E[t] (distinct content) × hand-object-interaction
        │                        importance → many 4.1–4.3 s segments → maximises distinct actions captured
        │
        └─► Sports branch (14 FPVSum categories)
               Highlights ...... importance = category "exciting moment" prompts − dull-frame prompts
                                 + motion + novelty + sharpness → best 4–12 s windows packed into 15 %
        ▼
 Constraint solver: sorted, strictly non-overlapping, every segment ≥ 4.1 s, total ≤ 15 %, 2-decimal safe
        ▼
 <video_id>.json  (team_id, video_id, duration_sec, label, actions, highlights)
```

| Metric (weight) | What we optimise | Component |
|---|---|---|
| Action coverage (40 %) | number of **distinct** GT actions inside highlights | short segments chosen for *diversity* (facility location) weighted by hand–object interaction |
| Action mAP (25 %) | ranked verb–noun detections at tIoU 0.1–0.5 | zero-shot pair scoring + train-split pair prior, multi-scale proposals, NMS, absolute (cross-video comparable) scores |
| Highlight F1 (25 %) | overlap with top-15 % human importance frames | semantic "exciting moment" prompts + motion + novelty + quality, contiguous windows |
| Label accuracy (10 %) | FPVSum category | prompt ensembles (3 descriptions × 3 templates per class) |

**Why zero-shot:** the evaluation videos *are* the EPIC-KITCHENS-100 validation split and the annotated FPVSum videos,
so we train nothing on them. A strong open vision-language model plus the public EPIC-100 **train** statistics
generalises without touching evaluation labels and runs on a single free T4 GPU.

## 2. Setup

Requirements: Linux, Python ≥ 3.10, `ffmpeg`/`ffprobe`, an NVIDIA GPU recommended (CPU works, slower).

```bash
pip install -r requirements.txt
python weights/download_weights.py        # downloads google/siglip-so400m-patch14-384 (~3.5 GB) into weights/hf/
                                          # and caches all text embeddings; inference is then offline
echo "CollegeName_TeamName" > team_id.txt
./run.sh path/to/P02_109.MP4 outputs/P02_109.json
python validate.py --file outputs/P02_109.json -v
```

Docker alternative:

```bash
docker build -t avsum .
docker run --gpus all -v $PWD/data:/data avsum /data/P02_109.MP4 /data/P02_109.json
```

## 3. Reproducing all 187 outputs

The easiest path is the Colab notebook **`notebooks/colab_pipeline.ipynb`** (free T4): it installs everything,
downloads the model, runs FPVSum, streams the 138 EPIC videos (download → process → delete, because ~184 GB does not fit
on Colab's disk), validates and zips the submission into Google Drive. Every step resumes after a disconnect.

Manual equivalent:

```bash
# FPVSum (49 videos, YouTube; use --cookies cookies.txt if YouTube asks for sign-in)
python code/download_fpvsum.py --out-dir videos/fpvsum
python code/batch.py --video-dir videos/fpvsum --dataset fpvsum --out-dir outputs

# EPIC-KITCHENS-100 validation (138 videos)
git clone https://github.com/epic-kitchens/epic-kitchens-download-scripts.git
python code/stream_epic.py --downloader-dir epic-kitchens-download-scripts --dl-dir /tmp/epic --out-dir outputs
# (or, if the videos are already on disk)
python code/batch.py --video-dir /path/to/EPIC-KITCHENS --dataset epic_kitchens_100 --out-dir outputs

python validate.py --dir outputs/
```

Useful flags (all scripts): `--fps 2.0`, `--model <hf id>` (e.g. `openai/clip-vit-large-patch14` for a lighter model),
`--hwaccel auto|cuda|none`, `--keyframes auto|on|off`, `--feat-cache DIR` (re-runs skip decoding), `--batch-size 32`.

## 4. Output contract

Matches `TEAM_README.md` exactly; `validate.py` re-implements every documented check (the official script was not in our
copy of the student kit) and `infer.py` validates each record before writing it.

- `duration_sec` from `ffprobe`; the official manifest is consulted **only** as a sanity check (if the container differs
  by > 0.3 s, the manifest value is used so the ±0.5 s rule holds).
- `label.name` ∈ the 16 official labels (we never predict `other`: every evaluation video is kitchen or one of 14 sports).
- `actions`: ≤ 2000 items, verbs/nouns from `supported_classes.json`, segments inside `[0, duration]`, scores in [0, 1].
  Empty for non-kitchen videos (allowed for FPVSum).
- `highlights`: chronologically sorted, strictly non-overlapping (≥ 0.3 s gaps), every segment ≥ 4.1 s (survives the
  4.0 s rule after rounding), total ≤ 15 % (videos shorter than ~29 s get a single 4.1 s segment, the minimum allowed).

## 5. Integrity

- **No evaluation ground truth is used anywhere.** EPIC-100 *validation* annotations and FPVSum `gt_data.mat` are never read.
- `weights/epic_action_prior.json` is built by `code/build_prior.py` from the public **EPIC_100_train.csv** only
  (verb–noun pair counts and median durations).
- The label is predicted from pixels; file names / video IDs are never used for prediction (the ID is only echoed into
  `video_id`). No video IDs are hard-coded.
- Deterministic: same video + same weights → same JSON.

## 6. Runtime (T4, fp16)

| Stage | Cost |
|---|---|
| Model load + prompt cache | ~1 min once (≈ 3k prompts) |
| Encoding | ~2 fps × duration frames; ~60–80 frames/s on T4 |
| Decoding | key-frame mode when GOP ≤ 0.5 s, else NVDEC; CPU fallback |
| Selection + actions | < 2 s per video |

## 7. Repository layout

```
run.sh                  official entry point: ./run.sh <video.mp4> <out.json>
validate.py             output validator (all TEAM_README rules)
requirements.txt        Python dependencies (requirements-lock.txt = exact versions used for the submitted outputs)
Dockerfile              optional container
code/
  infer.py              single video → JSON (used by run.sh)
  batch.py              all manifest videos in a folder, resumable
  stream_epic.py        download → process → delete for EPIC-KITCHENS
  download_fpvsum.py    yt-dlp downloader for the 49 FPVSum videos
  build_prior.py        EPIC-100 train-split verb–noun prior
  avsum/                video.py · encoder.py · vocab.py · signals.py · highlights.py · actions.py · pipeline.py · config.py
  data/                 supported_classes.json, manifest.csv (official), fpvsum_videolist.csv (FPVSum repo)
weights/
  download_weights.py   fetches the SigLIP model + caches text embeddings
  epic_action_prior.json
outputs/                187 validated <video_id>.json files
notebooks/colab_pipeline.ipynb
```

## 8. Limitations and next steps

- Zero-shot verbs from single frames are the weakest part (mAP). Next step: temporal adapters trained on EPIC-100 train
  clips, or an egocentric video-language model (e.g. EgoVLP/LaViLa features) for verbs.
- Highlight importance weights are hand-set (no FPVSum labels used); a small ranker trained on *unlabeled* FPVSum
  videos with self-supervision could improve F1.
- Audio is ignored (annotators also saw video only).
