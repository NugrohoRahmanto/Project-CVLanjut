#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

QUEUE_LOG="${QUEUE_LOG:-serial_yoloworld_bdd10k.log}"

nohup bash -c '
set -euo pipefail

run_one() {
  local model="$1"
  local batch_size="$2"
  local tag="$3"

  echo "[$(date "+%Y-%m-%d %H:%M:%S")] START ${tag}: model=${model}, batch=${batch_size}"
  "'"$PYTHON_BIN"'" scripts/run_train_yoloworld_bdd10k.py \
    --data-yaml data/bdd10k/bdd10k.yaml \
    --model "${model}" \
    --output-dir runs/yoloworld_bdd10k \
    --experiment-name "yoloworld_bdd10k_finetune_${tag}" \
    --timestamp-output \
    --epochs 100 \
    --batch-size "${batch_size}" \
    --imgsz 640 \
    --lr0 1e-4 \
    --known-classes "pedestrian,rider,car,truck,bus,train,motorcycle,bicycle,traffic light,traffic sign" \
    --unknown-prompts "" \
    --no-use-zero-shot-unknown-model \
    --device 0 \
    --workers 8 \
    --amp
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] FINISH ${tag}"
}

run_one yolov8s-world.pt 48 yolo_s
run_one yolov8m-world.pt 32 yolo_m
run_one yolov8l-world.pt 16 yolo_l

echo "[$(date "+%Y-%m-%d %H:%M:%S")] SERIAL TRAINING QUEUE FINISHED"
' > "$QUEUE_LOG" 2>&1 &

echo "Serial YOLO-World training queue started in background."
echo "Queue log: $QUEUE_LOG"
echo "Monitor queue: tail -f $QUEUE_LOG"
echo "Monitor active run: tail -f training.log"
