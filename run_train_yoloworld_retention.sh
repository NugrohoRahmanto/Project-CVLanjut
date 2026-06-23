#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${MODEL:-yolov8s-world.pt}"
TEACHER_MODEL="${TEACHER_MODEL:-$MODEL}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
IMG_SIZE="${IMG_SIZE:-640}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-yoloworld_bdd10k_retention_yolo_s_3class}"

"$PYTHON_BIN" scripts/yoloworld_bdd10k_retention_pipeline.py \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model "$MODEL" \
  --teacher-model "$TEACHER_MODEL" \
  --output-dir runs/yoloworld_bdd10k_retention \
  --experiment-name "$EXPERIMENT_NAME" \
  --timestamp-output \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --unknown-prompts "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --pseudo-splits train \
  --pseudo-conf-thres 0.25 \
  --pseudo-iou-thres 0.7 \
  --pseudo-known-iou-suppress 0.30 \
  --pseudo-max-per-image 60 \
  --freeze-text-encoder \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --imgsz "$IMG_SIZE" \
  --lr0 0.00005 \
  --lrf 0.01 \
  --weight-decay 0.0005 \
  --momentum 0.937 \
  --warmup-epochs 3.0 \
  --optimizer auto \
  --device "$DEVICE" \
  --workers 8 \
  --seed 42 \
  --amp \
  --patience 50 \
  --save-period -1 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --eval-split val \
  --research-eval
