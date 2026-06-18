#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

QUEUE_LOG="${QUEUE_LOG:-serial_yolo_bdd10k.log}"
SOURCE_DATA_YAML="${SOURCE_DATA_YAML:-data/bdd10k/bdd10k.yaml}"
THREE_CLASS_ROOT="${THREE_CLASS_ROOT:-data/bdd10k_yolo_3class}"

nohup bash -c '
set -euo pipefail

PYTHON_BIN="'"$PYTHON_BIN"'"
SOURCE_DATA_YAML="'"$SOURCE_DATA_YAML"'"
THREE_CLASS_ROOT="'"$THREE_CLASS_ROOT"'"

prepare_three_class_dataset() {
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] PREPARE 3-class YOLO dataset: ${THREE_CLASS_ROOT}"
  "$PYTHON_BIN" - <<'"'"'PY'"'"'
from pathlib import Path
import shutil
import yaml

source_yaml = Path("'"$SOURCE_DATA_YAML"'")
target_root = Path("'"$THREE_CLASS_ROOT"'")
data = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
source_root = Path(data.get("path", "."))
names = {int(k): str(v) for k, v in data["names"].items()}
target_names = ["car", "bus", "truck"]
old_to_new = {old_id: target_names.index(name) for old_id, name in names.items() if name in target_names}

if set(target_names) != {names[old_id] for old_id in old_to_new}:
    raise RuntimeError(f"Could not find all target classes {target_names} in {source_yaml}")

target_root.mkdir(parents=True, exist_ok=True)
for split in ("train", "val"):
    image_src = source_root / data[split]
    image_dst = target_root / "images" / split
    label_src = source_root / "labels" / split
    label_dst = target_root / "labels" / split
    if not image_src.exists():
        raise RuntimeError(f"Missing image source: {image_src}")
    if not label_src.exists():
        raise RuntimeError(f"Missing label source: {label_src}")
    if image_dst.exists() or image_dst.is_symlink():
        if image_dst.is_symlink() or image_dst.is_file():
            image_dst.unlink()
        else:
            shutil.rmtree(image_dst)
    image_dst.parent.mkdir(parents=True, exist_ok=True)
    image_dst.symlink_to(image_src.resolve(), target_is_directory=True)
    if label_dst.exists():
        shutil.rmtree(label_dst)
    label_dst.mkdir(parents=True, exist_ok=True)
    kept_images = 0
    kept_boxes = 0
    for label_file in sorted(label_src.glob("*.txt")):
        rows = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            old_id = int(float(parts[0]))
            if old_id not in old_to_new:
                continue
            rows.append(" ".join([str(old_to_new[old_id]), *parts[1:]]))
        (label_dst / label_file.name).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        if rows:
            kept_images += 1
            kept_boxes += len(rows)
    print(f"{split}: labels={label_dst} images_with_target={kept_images} boxes={kept_boxes}")

config = {
    "path": str(target_root),
    "train": "images/train",
    "val": "images/val",
    "names": {idx: name for idx, name in enumerate(target_names)},
}
(target_root / "bdd10k_3class.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"wrote {target_root / 'bdd10k_3class.yaml'}")
PY
}

run_one() {
  local data_yaml="$1"
  local model="$2"
  local batch_size="$3"
  local tag="$4"

  echo "[$(date "+%Y-%m-%d %H:%M:%S")] START ${tag}: data=${data_yaml}, model=${model}, batch=${batch_size}"
  "$PYTHON_BIN" scripts/run_train_yolo_bdd10k.py \
    --data-yaml "${data_yaml}" \
    --model "${model}" \
    --output-dir runs/yolo_bdd10k \
    --experiment-name "${tag}" \
    --timestamp-output \
    --epochs 100 \
    --batch-size "${batch_size}" \
    --imgsz 640 \
    --lr0 1e-4 \
    --device 0 \
    --workers 8 \
    --amp
  echo "[$(date "+%Y-%m-%d %H:%M:%S")] FINISH ${tag}"
}

prepare_three_class_dataset

run_one "${THREE_CLASS_ROOT}/bdd10k_3class.yaml" yolov8s.pt 48 yolo_bdd10k_3class_s
run_one "${THREE_CLASS_ROOT}/bdd10k_3class.yaml" yolov8m.pt 32 yolo_bdd10k_3class_m
run_one "${THREE_CLASS_ROOT}/bdd10k_3class.yaml" yolov8l.pt 16 yolo_bdd10k_3class_l

run_one "${SOURCE_DATA_YAML}" yolov8s.pt 48 yolo_bdd10k_10class_s
run_one "${SOURCE_DATA_YAML}" yolov8m.pt 32 yolo_bdd10k_10class_m
run_one "${SOURCE_DATA_YAML}" yolov8l.pt 16 yolo_bdd10k_10class_l

echo "[$(date "+%Y-%m-%d %H:%M:%S")] SERIAL YOLO TRAINING QUEUE FINISHED"
' > "$QUEUE_LOG" 2>&1 &

echo "Serial YOLO training queue started in background."
echo "Queue log: $QUEUE_LOG"
echo "Monitor queue: tail -f $QUEUE_LOG"
echo "Monitor active run: tail -f training.log"
