#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" scripts/run_pretrained_yoloworld_bdd10k.py \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --output-dir outputs/pretrained_yoloworld_bdd10k \
  --models yolov8s-world.pt,yolov8m-world.pt,yolov8l-world.pt \
  "$@"
