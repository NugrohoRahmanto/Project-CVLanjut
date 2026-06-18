#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

if [[ "$#" -eq 0 ]]; then
  set -- \
    --data-yaml data/bdd10k/bdd10k.yaml \
    --model yolov8s.pt \
    --output-dir runs/yolo_bdd10k \
    --experiment-name yolo_bdd10k_finetune \
    --timestamp-output \
    --epochs 50 \
    --batch-size 8 \
    --imgsz 640 \
    --lr0 1e-4 \
    --device 0 \
    --workers 8 \
    --amp
fi

nohup "$PYTHON_BIN" scripts/run_train_yolo_bdd10k.py "$@" >/dev/null 2>&1 &

echo "YOLO BDD10K run started in background."
echo "Monitor: tail -f training.log"
