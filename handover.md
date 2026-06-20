# Handover Codex - Project-CVLanjut

Last updated: 2026-06-20

## Project Goal

Riset membandingkan YOLO-World dan Standard YOLO pada BDD10K untuk skenario open-vocabulary/open-set detection.

Skema utama:

- YOLO-World all-class training: 10 class.
- YOLO-World known-class training: 3 class known, unknown dievaluasi zero-shot.
- Standard YOLO all-class training: 10 class.
- Standard YOLO known-class training: 3 class known, unknown dievaluasi sebagai limitasi closed-set.

Known classes resmi:

```text
car,bus,truck
```

Unknown classes resmi:

```text
pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign
```

YAML canonical class order:

```text
0 pedestrian
1 rider
2 car
3 truck
4 bus
5 train
6 motorcycle
7 bicycle
8 traffic light
9 traffic sign
```

Official eval command:

```bash
bash run_results_eval.sh \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --output-dir outputs/results_eval_bdd10k_official_prompt_fixed_labels \
  --device 0 \
  --batch-size 8 \
  --allow-failures
```

This is eval-only. It does not retrain checkpoints.

## Important Dataset Discovery

BDD10K raw JSON category names differ from the project YAML:

```text
person -> pedestrian
motor  -> motorcycle
bike   -> bicycle
```

This caused earlier YOLO labels to miss `pedestrian`, `motorcycle`, and `bicycle`.

Fixed in:

```text
scripts/convert_bdd10k_to_yolo.py
```

Relevant changes:

- Added `BDD10K_CATEGORY_ALIASES`.
- `convert_records()` now normalizes raw JSON category names before writing YOLO labels.

After regenerating labels, local val distribution from `data/bdd10k/labels/val` is:

```text
pedestrian: 871
rider: 39
car: 5141
truck: 246
bus: 88
train: 0
motorcycle: 20
bicycle: 67
traffic light: 1191
traffic sign: 1763
```

Important: raw `bdd100k_labels_images_val.json` has 10,000 records, but local available `data/bdd10k/images/val` contains only 454 images. Converter logs showed:

```text
val: converted=454 missing_images=9546
```

Therefore `train` class exists in full JSON val (15 objects / 14 images), but not in the local val subset used by Ultralytics. Per-class AP for `train` is N/A in local research eval.

## Evaluation Semantics Agreed In Chat

All models should run through comparable research evaluation, even if not fully fair:

- Evaluate all-class target.
- Evaluate unknown-class-only target.
- Keep pipeline behavior consistent across YOLO-World and Standard YOLO.

Important final interpretation:

### Standard YOLO known-class / 3-class

It should not be all zeros for all-class.

Current behavior:

- Known classes `car,bus,truck` are evaluated normally on a generated `known_class` dataset.
- All-class summary is macro-adjusted: known classes use real metrics, classes outside closed-set head are counted as zero if GT exists, N/A if no GT.
- Unknown-class summary remains zero because Standard YOLO is closed-set and has no semantic unknown head.

Example current CSV values:

```text
YOLO-S 3class all_mAP50: 0.1593818448
YOLO-M 3class all_mAP50: 0.1656203476
YOLO-L 3class all_mAP50: 0.1676059334
unknown_mAP50: 0.0 for all Standard YOLO 3class
```

### YOLO-World known-class / 3-class

There are two relevant branches:

- Known/all branch: fine-tuned 3-class checkpoint for known classes.
- Unknown branch: zero-shot model, usually `yolov8s-world.pt`, with unknown-only prompts and lower confidence threshold.

The final agreed behavior:

- In `unknown_class`, evaluate unknown classes using zero-shot branch.
- In `all_class` summary for YOLO-World known-class, synthesize/merge:
  - known class per-class metrics from the fine-tuned checkpoint,
  - unknown class per-class metrics from the `unknown_class` branch.

This makes per-class unknown metrics consistent between 7-class unknown table and 10-class all-class table.

Implemented in:

```text
scripts/research_evaluation.py
```

Key functions/logic:

- `build_research_eval_datasets()` now creates `all_class`, `unknown_class`, and `known_class` eval datasets.
- `extract_per_class_metrics()` maps Ultralytics AP arrays using `box.ap_class_index`, fixing wrong class placement in table 3.
- `expand_known_summary_to_all_classes()` handles Standard YOLO 3-class all-class macro adjustment.
- `merge_known_and_unknown_summary()` merges YOLO-World known branch and unknown branch for all-class summary.
- After both modes are evaluated, YOLO-World known-class all-class summary is replaced with the merged summary.

## Current Official Output

Main output directory:

```text
outputs/results_eval_bdd10k_official_prompt_fixed_labels
```

Important files:

```text
outputs/results_eval_bdd10k_official_prompt_fixed_labels/research_metrics_summary.csv
outputs/results_eval_bdd10k_official_prompt_fixed_labels/research_metrics_summary.json
outputs/results_eval_bdd10k_official_prompt_fixed_labels/table1_all_class.md
outputs/results_eval_bdd10k_official_prompt_fixed_labels/table2_zero_shot_known_train.md
outputs/results_eval_bdd10k_official_prompt_fixed_labels/table3_unknown_per_class_map50.md
outputs/results_eval_bdd10k_official_prompt_fixed_labels/table4_standard_yolo_known_analysis.md
outputs/results_eval_bdd10k_official_prompt_fixed_labels/table5_efficiency_latency.md
```

Latest selected CSV rows:

```text
yoloworld,all_class,s:   all_mAP50=0.2762676172, unknown_mAP50=0.1689024828
yoloworld,all_class,m:   all_mAP50=0.3082473581, unknown_mAP50=0.2142821499
yoloworld,all_class,l:   all_mAP50=0.3350334151, unknown_mAP50=0.2372962187

yoloworld,known_class,s: all_mAP50=0.2801646940, unknown_mAP50=0.1966550387
yoloworld,known_class,m: all_mAP50=0.3059337582, unknown_mAP50=0.1966550387
yoloworld,known_class,l: all_mAP50=0.3034673187, unknown_mAP50=0.1966550387

yolo,all_class,s:        all_mAP50=0.2854968794, unknown_mAP50=0.1906840437
yolo,all_class,m:        all_mAP50=0.3116589341, unknown_mAP50=0.2113769007
yolo,all_class,l:        all_mAP50=0.3176988596, unknown_mAP50=0.2142798308

yolo,known_class,s:      all_mAP50=0.1593818448, unknown_mAP50=0.0
yolo,known_class,m:      all_mAP50=0.1656203476, unknown_mAP50=0.0
yolo,known_class,l:      all_mAP50=0.1676059334, unknown_mAP50=0.0
```

For YOLO-World S known-class, per-class unknown now matches between `all_class` and `unknown_class`:

```text
pedestrian:    0.2220374285
rider:         0.0
train:         N/A
motorcycle:    0.2850000000
bicycle:       0.2742665717
traffic light: 0.1709556638
traffic sign:  0.2276705680
```

## Important Analytical Notes

Do not compare old `runs/.../metrics/evaluation.json` directly with the latest `outputs/.../research_metrics_summary.csv` without caveat.

Reason:

- `metrics/evaluation.json` was generated during training before alias fix.
- It used labels missing `pedestrian`, `motorcycle`, and `bicycle`.
- Latest research eval uses fixed labels and therefore has more GT objects.

Example:

```text
runs/yolo_bdd10k/20260619_001031_yolo_bdd10k_finetune_yolo_10class_l/metrics/evaluation.json
mAP50 = 0.4765494907

outputs/results_eval_bdd10k_official_prompt_fixed_labels/research_metrics_summary.csv
same run all_mAP50 = 0.3176988596
```

This difference is expected because the latest eval includes fixed GT for classes that were missing before.

## Prior Manual Calculations

Known 3-class mAP50 for YOLO-World S:

```text
YOLO-World S 10-class on car,bus,truck:
car=0.673903, bus=0.432435, truck=0.352878
macro=0.486406

YOLO-World S 3-class on car,bus,truck:
car=0.661740, bus=0.332841, truck=0.346971
macro=0.447184
```

Unknown 7-class mAP50 for YOLO-World S, present GT only:

```text
YOLO-World S 10-class: 0.168902
YOLO-World S 3-class:  0.196655
```

10-class all-class mAP50 for YOLO-World S after final merge behavior:

```text
YOLO-World S 10-class: 0.276268
YOLO-World S 3-class:  0.280165
```

The YOLO-World S 3-class all-class metric is now a merged summary:

- known class metrics from fine-tuned 3-class checkpoint,
- unknown class metrics from unknown-only zero-shot branch.

## Key Files Added Or Modified

Added or heavily updated:

```text
scripts/research_evaluation.py
scripts/evaluate_research_metrics.py
scripts/run_results_eval.py
scripts/visualize_research_eval.py
run_evaluate_research_metrics.sh
run_results_eval.sh
results.md
```

Modified:

```text
scripts/convert_bdd10k_to_yolo.py
README.md
run_train_yoloworld_bdd10k_serial.sh
scripts/yolo_bdd10k_pipeline.py
scripts/yoloworld_bdd10k_pipeline.py
```

Current git status at handover time included modified/untracked files:

```text
 M README.md
 M run_train_yoloworld_bdd10k_serial.sh
 M scripts/convert_bdd10k_to_yolo.py
 M scripts/yolo_bdd10k_pipeline.py
 M scripts/yoloworld_bdd10k_pipeline.py
?? results.md
?? run_evaluate_research_metrics.sh
?? run_results_eval.sh
?? scripts/evaluate_research_metrics.py
?? scripts/research_evaluation.py
?? scripts/run_results_eval.py
?? scripts/visualize_research_eval.py
```

This handover file itself may also be untracked after creation.

## Commands To Verify State

Check fixed dataset distribution:

```bash
.venv/bin/python scripts/check_bdd10k_dataset.py --data-yaml data/bdd10k/bdd10k.yaml
```

Regenerate BDD10K labels with aliases:

```bash
.venv/bin/python scripts/convert_bdd10k_to_yolo.py \
  --data-root data/bdd10k \
  --overwrite
```

Run official research eval:

```bash
bash run_results_eval.sh \
  --known-classes "car,bus,truck" \
  --unknown-classes "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign" \
  --output-dir outputs/results_eval_bdd10k_official_prompt_fixed_labels \
  --device 0 \
  --batch-size 8 \
  --allow-failures
```

Visualize research eval for one run:

```bash
.venv/bin/python scripts/visualize_research_eval.py \
  --run-dir runs/yoloworld_bdd10k/20260617_214915_yoloworld_bdd10k_finetune_yolo_l_3class \
  --mode both \
  --sample-count 12 \
  --device 0
```

## Open Caveats

1. Local val subset has only 454 images, not full BDD10K val 10,000 images.
2. `train` class has 0 GT in local val, so per-class AP for `train` is N/A.
3. Existing 10-class checkpoints were trained before alias fix, so supervised performance on `pedestrian`, `motorcycle`, and `bicycle` is likely not valid as a fully corrected supervised result.
4. YOLO-World 3-class unknown results use zero-shot `yolov8s-world.pt`, so S/M/L known-class unknown metrics are identical by design unless the zero-shot model setting is changed.
5. `table4_standard_yolo_known_analysis.md` is still a placeholder-style analysis table unless a more detailed class-agnostic IoU analysis is implemented later.

## Suggested Next Steps

1. Decide whether final paper uses local 454-image val or obtains full 10,000-image BDD10K val.
2. If final supervised 10-class claims are required, retrain 10-class models after alias fix.
3. If keeping current checkpoints, clearly state that latest research eval uses corrected labels, while old training-time eval used pre-fix labels.
4. Optionally implement real Table 4 class-agnostic recall/misclassification/miss-rate for Standard YOLO known-class models.
