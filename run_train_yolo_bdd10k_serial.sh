#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

QUEUE_LOG="${QUEUE_LOG:-serial_yolo_bdd10k.log}"

nohup bash -c '
set -euo pipefail

run_one() {
  local model="$1"
  local batch_size="$2"
  local tag="$3"
  shift 3

  echo "[$(date "+%Y-%m-%d %H:%M:%S")] START ${tag}: model=${model}, batch=${batch_size}"
  "'"$PYTHON_BIN"'" scripts/run_train_yolo_bdd10k.py \
    --data-yaml data/bdd10k/bdd10k.yaml \
    --model "${model}" \
    --output-dir runs/yolo_bdd10k \
    --experiment-name "yolo_bdd10k_finetune_${tag}" \
    --timestamp-output \
    --epochs 100 \
    --batch-size "${batch_size}" \
    --imgsz 640 \
    --lr0 1e-4 \
    --device 0 \
    --workers 8 \
    --amp \
    "$@"
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] FINISH ${tag}"
}

echo "[$(date "+%Y-%m-%d %H:%M:%S")] SERIAL GROUP START: YOLO standard 3-class default"
run_one yolov8s.pt 48 yolo_3class_s
run_one yolov8m.pt 32 yolo_3class_m
run_one yolov8l.pt 16 yolo_3class_l

echo "[$(date "+%Y-%m-%d %H:%M:%S")] SERIAL GROUP START: YOLO standard 10-class full"
run_one yolov8s.pt 48 yolo_10class_s --skip-filtered-dataset
run_one yolov8m.pt 32 yolo_10class_m --skip-filtered-dataset
run_one yolov8l.pt 16 yolo_10class_l --skip-filtered-dataset

echo "[$(date "+%Y-%m-%d %H:%M:%S")] SERIAL YOLO STANDARD TRAINING QUEUE FINISHED"
' > "$QUEUE_LOG" 2>&1 &

echo "Serial YOLO standard training queue started in background."
echo "Queue log: $QUEUE_LOG"
echo "Monitor queue: tail -f $QUEUE_LOG"
echo "Monitor active run: tail -f training.log"
