from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from research_evaluation import (
    find_images,
    infer_model_type,
    load_model,
    normalize_names,
    run_dir_checkpoint,
    set_yoloworld_classes,
)


KNOWN_PRED_COLOR = (0, 170, 70)
ZERO_SHOT_PRED_COLOR = (185, 45, 185)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create GT vs prediction visualizations for research_eval outputs.")
    parser.add_argument("--run-dir", required=True, help="Experiment directory containing weights and research_eval.")
    parser.add_argument("--research-dir", default="research_eval")
    parser.add_argument("--mode", choices=("all_class", "unknown_class", "both"), default="both")
    parser.add_argument("--model-type", choices=("auto", "yolo", "yoloworld"), default="auto")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--filter-pred-to-gt-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/<research-dir>/visualizations")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_research_info(research_root: Path) -> dict[str, Any]:
    path = research_root / "datasets" / "research_eval_datasets.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing research eval dataset metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_research_metrics(research_root: Path) -> dict[str, Any]:
    path = research_root / "research_metrics_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label_path_for_image(image_path: Path, dataset_root: Path, split: str) -> Path:
    image_root = dataset_root / "images" / split
    return dataset_root / "labels" / split / image_path.relative_to(image_root).with_suffix(".txt")


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = values
    return [
        (xc - bw / 2.0) * width,
        (yc - bh / 2.0) * height,
        (xc + bw / 2.0) * width,
        (yc + bh / 2.0) * height,
    ]


def draw_box(image: Any, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    label_y = max(y1, th + baseline + 4)
    cv2.rectangle(image, (x1, label_y - th - baseline - 4), (min(x1 + tw + 6, w - 1), label_y + baseline), color, -1)
    cv2.putText(image, label, (x1 + 3, label_y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def add_title(image: Any, title: str) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_ground_truth(image: Any, image_path: Path, dataset_root: Path, split: str, names: dict[int, str]) -> tuple[int, set[int]]:
    label_path = label_path_for_image(image_path, dataset_root, split)
    height, width = image.shape[:2]
    count = 0
    class_ids: set[int] = set()
    if not label_path.exists():
        return count, class_ids
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        class_ids.add(class_id)
        box = yolo_to_xyxy([float(value) for value in parts[1:]], width, height)
        draw_box(image, box, names.get(class_id, str(class_id)), (255, 130, 0))
        count += 1
    return count, class_ids


def collect_label_class_ids(dataset_root: Path, split: str) -> set[int]:
    class_ids: set[int] = set()
    label_root = dataset_root / "labels" / split
    if not label_root.exists():
        return class_ids
    for label_path in label_root.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                class_ids.add(int(float(parts[0])))
    return class_ids


def sample_images(dataset_yaml: Path, sample_count: int, start_index: int) -> tuple[list[Path], Path, str, dict[int, str], set[int]]:
    config = load_yaml(dataset_yaml)
    dataset_root = Path(config.get("path", dataset_yaml.parent))
    split = "val"
    image_root = dataset_root / config.get(split, "images/val")
    names = normalize_names(config.get("names", {}))
    label_class_ids = collect_label_class_ids(dataset_root, split)
    images = find_images(image_root)
    if start_index > 0:
        images = images[start_index:]
    return images[:sample_count], dataset_root, split, names, label_class_ids


def class_id_map(source_order: list[str], target_names: dict[int, str]) -> dict[int, int]:
    target_by_name = {name: class_id for class_id, name in target_names.items()}
    return {source_id: target_by_name[name] for source_id, name in enumerate(source_order) if name in target_by_name}


def draw_predictions(
    image: Any,
    result: Any,
    names: dict[int, str],
    label_class_ids: set[int],
    mode: str,
    filter_pred_to_gt_classes: bool,
    class_map: dict[int, int] | None = None,
    color: tuple[int, int, int] = (0, 170, 70),
) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    pred_count = 0
    for box, conf_value, class_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
        source_class_id = int(class_value)
        class_id = class_map.get(source_class_id, source_class_id) if class_map else source_class_id
        if filter_pred_to_gt_classes and mode == "unknown_class" and class_id not in label_class_ids:
            continue
        label = f"{names.get(class_id, str(class_id))} {float(conf_value):.2f}"
        draw_box(image, [float(value) for value in box], label, color)
        pred_count += 1
    return pred_count


def move_model_to_predict_device(model: Any, device: str) -> None:
    if not device or device == "cpu":
        return
    module = getattr(model, "model", None)
    if module is not None and hasattr(module, "to"):
        module.to(f"cuda:{device}" if str(device).isdigit() else device)


def visualize_mode(
    mode: str,
    dataset_yaml: Path,
    output_root: Path,
    sample_count: int,
    start_index: int,
    iou: float,
    imgsz: int,
    device: str,
    filter_pred_to_gt_classes: bool,
    prediction_branches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import cv2

    images, dataset_root, split, names, label_class_ids = sample_images(dataset_yaml, sample_count, start_index)
    if not images:
        raise RuntimeError(f"No images found for visualization dataset: {dataset_yaml}")
    output_dir = output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_predictions = []
    for branch in prediction_branches:
        move_model_to_predict_device(branch["model"], device)
        branch_predictions.append(
            {
                **branch,
                "predictions": branch["model"].predict(
                    source=[str(path) for path in images],
                    conf=branch["conf"],
                    iou=iou,
                    imgsz=imgsz,
                    device=device,
                    save=False,
                    verbose=False,
                ),
            }
        )
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images):
        original = cv2.imread(str(image_path))
        if original is None:
            continue
        gt_panel = original.copy()
        pred_panel = original.copy()
        gt_count, _ = draw_ground_truth(gt_panel, image_path, dataset_root, split, names)
        pred_count = 0
        branch_counts: dict[str, int] = {}
        for branch in branch_predictions:
            predictions = branch["predictions"]
            result = predictions[index] if index < len(predictions) else None
            count = draw_predictions(
                image=pred_panel,
                result=result,
                names=names,
                label_class_ids=label_class_ids,
                mode=mode,
                filter_pred_to_gt_classes=filter_pred_to_gt_classes,
                class_map=branch.get("class_map"),
                color=branch.get("color", (0, 170, 70)),
            )
            branch_counts[str(branch.get("name", "prediction"))] = count
            pred_count += count
        add_title(gt_panel, f"Ground Truth {mode} ({gt_count})")
        add_title(pred_panel, f"Prediction ({pred_count})")
        combined = cv2.hconcat([gt_panel, pred_panel])
        out_path = output_dir / f"{image_path.stem}_gt_vs_pred.jpg"
        cv2.imwrite(str(out_path), combined)
        rows.append(
            {
                "mode": mode,
                "image": str(image_path),
                "output": str(out_path),
                "ground_truth_boxes": gt_count,
                "prediction_boxes": pred_count,
                "prediction_branch_boxes": branch_counts,
            }
        )
    return rows


def model_from_source(source_model: str, checkpoint: Path, model_type: str, default_model: Any, cache: dict[str, Any]) -> Any:
    if not source_model or source_model == str(checkpoint):
        return default_model
    if source_model not in cache:
        cache[source_model] = load_model(Path(source_model), model_type)
    return cache[source_model]


def resolve_visualization_checkpoint(run_dir: Path, checkpoint_name: str, research_metrics: dict[str, Any]) -> Path:
    try:
        return run_dir_checkpoint(run_dir, checkpoint_name)
    except FileNotFoundError:
        summary_checkpoint = (research_metrics.get("summary_row") or {}).get("checkpoint")
        if summary_checkpoint:
            return Path(str(summary_checkpoint))
        for mode_info in (research_metrics.get("metrics") or {}).values():
            source_model = mode_info.get("source_model") if isinstance(mode_info, dict) else None
            if source_model:
                return Path(str(source_model))
        raise


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    research_root = run_dir / args.research_dir
    research_info = load_research_info(research_root)
    research_metrics = load_research_metrics(research_root)
    training_config = str((research_metrics.get("summary_row") or {}).get("training_config", ""))
    modes = ["all_class", "unknown_class"] if args.mode == "both" else [args.mode]
    checkpoint = resolve_visualization_checkpoint(run_dir, args.checkpoint_name, research_metrics)
    model_type = infer_model_type(run_dir, args.model_type)
    model = load_model(checkpoint, model_type)
    model_cache: dict[str, Any] = {}
    output_root = Path(args.output_dir) if args.output_dir else research_root / "visualizations"
    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_info = research_info["datasets"][mode]
        metric_info = (research_metrics.get("metrics") or {}).get(mode, {})
        dataset_yaml = Path(mode_info["yaml"])
        class_order = mode_info.get("class_order") or research_info.get("evaluation_class_order") or []
        target_names = normalize_names(load_yaml(dataset_yaml).get("names", {}))
        prediction_branches: list[dict[str, Any]] = []

        merged_sources = metric_info.get("merged_sources") or {}
        if mode == "all_class" and model_type == "yoloworld" and merged_sources:
            known_info = research_info["datasets"].get("known_class") or {}
            unknown_info = research_info["datasets"].get("unknown_class") or {}
            known_order = list(known_info.get("class_order") or [])
            unknown_order = list(unknown_info.get("class_order") or [])
            known_source = (merged_sources.get("known") or {}).get("source_model", "")
            unknown_source = (merged_sources.get("unknown") or {}).get("source_model", "")
            known_model = model_from_source(str(known_source), checkpoint, model_type, model, model_cache)
            unknown_model = model_from_source(str(unknown_source), checkpoint, model_type, model, model_cache)
            if known_order:
                set_yoloworld_classes(known_model, known_order)
            if unknown_order:
                set_yoloworld_classes(unknown_model, unknown_order)
            prediction_branches.extend(
                [
                    {
                        "name": "known",
                        "model": known_model,
                        "conf": float((merged_sources.get("known") or {}).get("conf", args.conf_thres)),
                        "class_map": class_id_map(known_order, target_names),
                        "color": KNOWN_PRED_COLOR,
                    },
                    {
                        "name": "unknown",
                        "model": unknown_model,
                        "conf": float((merged_sources.get("unknown") or {}).get("conf", args.conf_thres)),
                        "class_map": class_id_map(unknown_order, target_names),
                        "color": ZERO_SHOT_PRED_COLOR,
                    },
                ]
            )
        else:
            mode_model = model
            mode_conf = args.conf_thres
            source_model = metric_info.get("source_model")
            if model_type == "yoloworld" and source_model and source_model != str(checkpoint):
                mode_model = model_from_source(str(source_model), checkpoint, model_type, model, model_cache)
                mode_conf = float(metric_info.get("conf", args.conf_thres))
            if model_type == "yoloworld" and class_order:
                set_yoloworld_classes(mode_model, list(class_order))
            prediction_branches.append(
                {
                    "name": mode,
                    "model": mode_model,
                    "conf": mode_conf,
                    "class_map": None,
                    "color": ZERO_SHOT_PRED_COLOR if training_config == "pretrained" else KNOWN_PRED_COLOR,
                }
            )
        rows.extend(
            visualize_mode(
                mode=mode,
                dataset_yaml=dataset_yaml,
                output_root=output_root,
                sample_count=args.sample_count,
                start_index=args.start_index,
                iou=args.iou_thres,
                imgsz=args.imgsz,
                device=args.device,
                filter_pred_to_gt_classes=args.filter_pred_to_gt_classes,
                prediction_branches=prediction_branches,
            )
        )
    summary_path = output_root / "visualization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved visualizations: {output_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
