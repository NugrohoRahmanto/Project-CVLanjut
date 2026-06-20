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


def visualize_mode(
    model: Any,
    mode: str,
    dataset_yaml: Path,
    output_root: Path,
    sample_count: int,
    start_index: int,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    filter_pred_to_gt_classes: bool,
) -> list[dict[str, Any]]:
    import cv2

    images, dataset_root, split, names, label_class_ids = sample_images(dataset_yaml, sample_count, start_index)
    if not images:
        raise RuntimeError(f"No images found for visualization dataset: {dataset_yaml}")
    output_dir = output_root / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(
        source=[str(path) for path in images],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,
        verbose=False,
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
        result = predictions[index] if index < len(predictions) else None
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box, conf_value, class_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                class_id = int(class_value)
                if filter_pred_to_gt_classes and mode == "unknown_class" and class_id not in label_class_ids:
                    continue
                label = f"{names.get(class_id, str(class_id))} {float(conf_value):.2f}"
                draw_box(pred_panel, [float(value) for value in box], label, (0, 170, 70))
                pred_count += 1
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
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    research_root = run_dir / args.research_dir
    research_info = load_research_info(research_root)
    research_metrics = load_research_metrics(research_root)
    modes = ["all_class", "unknown_class"] if args.mode == "both" else [args.mode]
    checkpoint = run_dir_checkpoint(run_dir, args.checkpoint_name)
    model_type = infer_model_type(run_dir, args.model_type)
    model = load_model(checkpoint, model_type)
    output_root = Path(args.output_dir) if args.output_dir else research_root / "visualizations"
    rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_info = research_info["datasets"][mode]
        metric_info = (research_metrics.get("metrics") or {}).get(mode, {})
        dataset_yaml = Path(mode_info["yaml"])
        class_order = mode_info.get("class_order") or research_info.get("evaluation_class_order") or []
        mode_model = model
        mode_conf = args.conf_thres
        source_model = metric_info.get("source_model")
        if model_type == "yoloworld" and source_model and source_model != str(checkpoint):
            mode_model = load_model(Path(source_model), model_type)
            mode_conf = float(metric_info.get("conf", args.conf_thres))
        if model_type == "yoloworld" and class_order:
            set_yoloworld_classes(mode_model, list(class_order))
        rows.extend(
            visualize_mode(
                model=mode_model,
                mode=mode,
                dataset_yaml=dataset_yaml,
                output_root=output_root,
                sample_count=args.sample_count,
                start_index=args.start_index,
                conf=mode_conf,
                iou=args.iou_thres,
                imgsz=args.imgsz,
                device=args.device,
                filter_pred_to_gt_classes=args.filter_pred_to_gt_classes,
            )
        )
    summary_path = output_root / "visualization_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved visualizations: {output_root}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
