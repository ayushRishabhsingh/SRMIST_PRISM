#!/usr/bin/env bash
# Official single-video entry point:   ./run.sh <video.mp4> <out.json>
# Optional env: TEAM_ID (or team_id.txt), AVSUM_MODEL (default google/siglip-so400m-patch14-384), PYTHON.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 2 ]; then
  echo "usage: ./run.sh <video.mp4> <out.json>" >&2
  exit 2
fi
exec "${PYTHON:-python3}" "$HERE/code/infer.py" "$1" "$2" "${@:3}"
