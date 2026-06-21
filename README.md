# Open-Vocabulary Hazard Perception for Autonomous Driving

Research pipeline for evaluating **YOLO-World** and **Standard YOLO** on BDD10K under autonomous-driving open-vocabulary object detection scenarios.

Recommended paper title:

```text
Open-Vocabulary Hazard Perception for Autonomous Driving: Evaluating YOLO-World Against Closed-Set YOLO on Unseen Road Objects
```

Alternative title ideas:

```text
Zero-Shot Road Object Detection for Autonomous Systems Using YOLO-World
Open-Vocabulary Object Detection for Autonomous Driving Safety on BDD10K
Evaluating Open-Vocabulary and Closed-Set YOLO Detectors for Autonomous Driving Perception
Unknown Object Detection in Autonomous Driving: A YOLO-World and Standard YOLO Study
```

## Research Scope

This project compares open-vocabulary and closed-set object detectors for autonomous driving. The central research question is whether YOLO-World can detect road objects that were not included during supervised fine-tuning, while Standard YOLO remains limited by its closed-set detection head.

Default BDD10K class split:

```text
Known classes   : car,bus,truck
Unknown classes : pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign
All classes     : pedestrian,rider,car,truck,bus,train,motorcycle,bicycle,traffic light,traffic sign
```

Main experiment schemes:

| Scheme | Model | Training Config | Purpose |
| --- | --- | --- | --- |
| 1 | YOLO-World | 10 classes | Open-vocabulary upper-bound after all-class fine-tuning |
| 2 | YOLO-World | 3 known classes | Zero-shot unknown-class detection through text prompts |
| 3 | Standard YOLO | 10 classes | Closed-set all-class supervised baseline |
| 4 | Standard YOLO | 3 known classes | Closed-set known-only limitation baseline |
| 5 | YOLO-World | Pure pretrained | No BDD10K training; full prompt-based zero-shot baseline |

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

The project uses Python `>=3.10,<3.11`. Shell wrappers automatically prefer `.venv/bin/python` when available.

## Dataset Preparation

Download and extract BDD10K into `data/bdd10k`, then convert labels into YOLO format:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --data-root data/bdd10k
```

Validate the converted dataset:

```bash
.venv/bin/python scripts/check_bdd10k_dataset.py \
  --data-yaml data/bdd10k/bdd10k.yaml
```

Expected canonical class order in `data/bdd10k/bdd10k.yaml`:

```yaml
names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
```

The converter normalizes BDD raw labels:

```text
person -> pedestrian
motor  -> motorcycle
bike   -> bicycle
```

## Training

Train YOLO-World:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model yolov8l-world.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name yoloworld_bdd10k_finetune_yolo_l_3class \
  --timestamp-output \
  --known-classes "car,bus,truck" \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --device 0 \
  --amp
```

Train Standard YOLO:

```bash
bash run_train_yolo_bdd10k.sh \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model yolov8l.pt \
  --output-dir runs/yolo_bdd10k \
  --experiment-name yolo_bdd10k_finetune_yolo_3class_l \
  --timestamp-output \
  --train-classes "car,bus,truck" \
  --epochs 100 \
  --batch-size 16 \
  --imgsz 640 \
  --device 0 \
  --amp
```

Serial training scripts are available for small, medium, and large variants:

```bash
bash run_train_yoloworld_bdd10k_serial.sh
bash run_train_yolo_bdd10k_serial.sh
```

## Research Evaluation

Research evaluation always creates two evaluation targets:

```text
all_class     : all 10 BDD10K classes
unknown_class : only the 7 unknown/unseen classes
```

Metrics:

```text
mAP50, mAP50-95, Precision, Recall, F1-Score
```

### Evaluate All Existing Trained Runs

```bash
bash run_results_eval.sh \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --output-dir outputs/results_eval_bdd10k_official \
  --device 0 \
  --batch-size 8 \
  --allow-failures
```

Outputs:

```text
outputs/results_eval_bdd10k_official/
  research_metrics_summary.csv
  research_metrics_summary.json
  table1_all_class.md
  table2_zero_shot_known_train.md
  table3_unknown_per_class_map50.md
  table4_standard_yolo_known_analysis.md
  table5_efficiency_latency.md
  table6_pretrained_yoloworld.md
```

### Evaluate One YOLO-World Large 3-Class Run

```bash
.venv/bin/python scripts/evaluate_research_metrics.py \
  --run-dir runs/yoloworld_bdd10k/20260617_214915_yoloworld_bdd10k_finetune_yolo_l_3class \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model-type yoloworld \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --output-name research_eval \
  --summary-out runs/yoloworld_l_3class_research_summary.csv \
  --split val \
  --imgsz 640 \
  --batch-size 8 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --device 0 \
  --workers 8
```

### Evaluate One Standard YOLO Large 3-Class Run

```bash
.venv/bin/python scripts/evaluate_research_metrics.py \
  --run-dir runs/yolo_bdd10k/20260618_214703_yolo_bdd10k_finetune_yolo_3class_l \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --model-type yolo \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --output-name research_eval \
  --summary-out runs/yolo_l_3class_research_summary.csv \
  --split val \
  --imgsz 640 \
  --batch-size 8 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --device 0 \
  --workers 8
```

## Pretrained Baselines

### Pure Pretrained YOLO-World

This baseline runs YOLO-World small, medium, and large from pretrained Ultralytics weights only. No BDD10K training or fine-tuning is performed.

```bash
bash run_pretrained_yoloworld_bdd10k.sh \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --device 0 \
  --batch-size 8
```

Outputs:

```text
outputs/pretrained_yoloworld_bdd10k/
  pretrained_yoloworld_summary.csv
  pretrained_yoloworld_summary.json
  table6_pretrained_yoloworld.md
  table7_pretrained_yoloworld_unknown_per_class_map50.md
  table8_pretrained_yoloworld_efficiency_latency.md
  yolov8s-world/research_eval/
  yolov8m-world/research_eval/
  yolov8l-world/research_eval/
```

For pretrained YOLO-World visualization, all prediction boxes are rendered purple because they are zero-shot prompt detections.

### Pure Pretrained Standard YOLO

Directly evaluating `yolov8s.pt`, `yolov8m.pt`, or `yolov8l.pt` on `bdd10k.yaml` is **not methodologically valid** for the official research tables because pretrained Standard YOLO uses COCO class IDs, while BDD10K uses a different 10-class ID order.

Use Standard YOLO pretrained weights as training initialization, then evaluate the BDD10K fine-tuned checkpoints with the research evaluator. This is the valid closed-set baseline used in this project.

For qualitative-only inspection of a raw COCO-pretrained YOLO model, use Ultralytics prediction directly and treat it as a non-comparable sanity check:

```bash
.venv/bin/python - <<'PY'
from ultralytics import YOLO

model = YOLO("yolov8l.pt")
model.predict(
    source="data/bdd10k/images/val",
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device=0,
    save=True,
    project="outputs/pretrained_yolo_bdd10k",
    name="yolov8l_coco_visual_only",
)
PY
```

Use this output only for visual reference, not for BDD10K mAP comparison.

## Visualization

The visualizer reads `research_eval` outputs and creates side-by-side panels:

```text
left  : ground truth boxes
right : prediction boxes
```

Color convention:

```text
orange : ground truth
green  : supervised known-class predictions
purple : zero-shot/open-vocabulary predictions
```

### Visualize YOLO-World Large 3-Class

```bash
.venv/bin/python scripts/visualize_research_eval.py \
  --run-dir runs/yoloworld_bdd10k/20260617_214915_yoloworld_bdd10k_finetune_yolo_l_3class \
  --research-dir research_eval \
  --mode both \
  --model-type yoloworld \
  --checkpoint-name best.pt \
  --sample-count 24 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --imgsz 640 \
  --device 0
```

YOLO-World 3-class `all_class` visualizations merge:

```text
known branch   : fine-tuned 3-class checkpoint
unknown branch : pretrained YOLO-World zero-shot model
```

### Visualize Standard YOLO Large 3-Class

```bash
.venv/bin/python scripts/visualize_research_eval.py \
  --run-dir runs/yolo_bdd10k/20260618_214703_yolo_bdd10k_finetune_yolo_3class_l \
  --research-dir research_eval \
  --mode both \
  --model-type yolo \
  --checkpoint-name best.pt \
  --sample-count 24 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --imgsz 640 \
  --device 0
```

For Standard YOLO 3-class, unknown objects cannot appear as true unknown labels because the detection head is closed-set and only contains `car`, `bus`, and `truck`.

### Visualize Pure Pretrained YOLO-World Large

```bash
.venv/bin/python scripts/visualize_research_eval.py \
  --run-dir outputs/pretrained_yoloworld_bdd10k/yolov8l-world \
  --research-dir research_eval \
  --mode both \
  --model-type yoloworld \
  --sample-count 24 \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --imgsz 640 \
  --device 0
```

Output location:

```text
<run-dir>/research_eval/visualizations/
  all_class/
  unknown_class/
  visualization_summary.json
```

## Prediction

YOLO-World prediction with known and unknown branches:

```bash
bash run_train_yoloworld_bdd10k.sh \
  --model runs/yoloworld_bdd10k/<run-name>/weights/best.pt \
  --output-dir runs/yoloworld_bdd10k \
  --experiment-name predict_yoloworld_bdd10k \
  --timestamp-output \
  --predict-only \
  --source data/bdd10k/images/val \
  --known-classes "car,bus,truck" \
  --unknown-prompts "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --zero-shot-unknown-model yolov8s-world.pt \
  --conf-thres 0.25 \
  --unknown-conf-thres 0.05 \
  --iou-thres 0.7 \
  --device 0
```

Standard YOLO prediction:

```bash
bash run_train_yolo_bdd10k.sh \
  --model runs/yolo_bdd10k/<run-name>/weights/best.pt \
  --output-dir runs/yolo_bdd10k \
  --experiment-name predict_yolo_bdd10k \
  --timestamp-output \
  --predict-only \
  --source data/bdd10k/images/val \
  --conf-thres 0.25 \
  --iou-thres 0.7 \
  --device 0
```

## Outputs and Logs

Trained YOLO-World runs:

```text
runs/yoloworld_bdd10k/<experiment-name>/
  configs/
  dataset_known/
  evaluation/
  logs/
  metrics/
  predictions/
  predictions_unknown/
  research_eval/
  weights/
```

Trained Standard YOLO runs:

```text
runs/yolo_bdd10k/<experiment-name>/
  configs/
  dataset_train_classes/
  evaluation/
  logs/
  metrics/
  predictions/
  research_eval/
  weights/
```

Monitor a background run:

```bash
tail -f training.log
```

Find latest runs:

```bash
ls -td runs/yoloworld_bdd10k/* | head -1
ls -td runs/yolo_bdd10k/* | head -1
```

## Troubleshooting

If CUDA is unavailable, change `--device 0` to `--device cpu` for a small test.

If ground-truth boxes are empty in visualization, regenerate labels:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --data-root data/bdd10k
```

If YOLO-World unknown boxes do not appear in a fine-tuned 3-class visualization, rerun research evaluation and visualization. The current visualizer merges `merged_sources` from `research_metrics_summary.json` so `all_class` can show both known and unknown predictions.

If Standard YOLO 3-class does not show unknown labels, that is expected. It is the intended closed-set limitation baseline.
