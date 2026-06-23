from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_KNOWN_CLASSES = ["car", "bus", "truck"]
DEFAULT_UNKNOWN_CLASSES = ["pedestrian", "rider", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError("names must be a dict or list")


def resolve_split_path(data_yaml: Path, config: dict[str, Any], split: str) -> Path:
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        candidate = data_yaml.parent / root
        root = candidate if candidate.exists() else root
    value = config.get(split)
    if not value:
        raise ValueError(f"Dataset yaml does not define split: {split}")
    split_path = Path(value)
    return split_path if split_path.is_absolute() else root / split_path


def label_root_from_image_root(image_root: Path) -> Path:
    text = str(image_root)
    token = f"{os.sep}images{os.sep}"
    if token in text:
        return Path(text.replace(token, f"{os.sep}labels{os.sep}"))
    return image_root.parent.parent / "labels" / image_root.name


def label_path_for_image(image: Path, image_root: Path, label_root: Path) -> Path:
    return label_root / image.relative_to(image_root).with_suffix(".txt")


def find_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def class_order_for_evaluation(
    source_names: list[str],
    trained_classes: list[str],
    known_classes: list[str],
    unknown_classes: list[str],
) -> list[str]:
    if trained_classes and len(trained_classes) == len(source_names) and set(trained_classes) == set(source_names):
        return trained_classes
    ordered = list(trained_classes or known_classes)
    for name in unknown_classes:
        if name not in ordered:
            ordered.append(name)
    for name in source_names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def infer_unknown_classes(source_names: list[str], known_classes: list[str], unknown_classes: list[str] | None = None) -> list[str]:
    if unknown_classes:
        return unknown_classes
    known = set(known_classes)
    return [name for name in source_names if name not in known]


def build_research_eval_datasets(
    source_data_yaml: Path,
    output_root: Path,
    trained_classes: list[str],
    known_classes: list[str] | None = None,
    unknown_classes: list[str] | None = None,
    split: str = "val",
    unknown_class_prompt_only: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    config = load_yaml(source_data_yaml)
    source_names_by_id = normalize_names(config.get("names", {}))
    source_names = [source_names_by_id[idx] for idx in sorted(source_names_by_id)]
    known = known_classes or DEFAULT_KNOWN_CLASSES
    unknown = infer_unknown_classes(source_names, known, unknown_classes)
    eval_order = class_order_for_evaluation(source_names, trained_classes, known, unknown)
    unknown_source_ids = {idx for idx, name in source_names_by_id.items() if name in set(unknown)}

    image_root = resolve_split_path(source_data_yaml, config, split)
    label_root = label_root_from_image_root(image_root)
    images = find_images(image_root)
    if not images:
        raise RuntimeError(f"No images found for research evaluation split={split}: {image_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_info: dict[str, Any] = {
        "source_data_yaml": str(source_data_yaml),
        "source_split": split,
        "source_image_root": str(image_root),
        "source_label_root": str(label_root),
        "trained_classes": trained_classes,
        "known_classes": known,
        "unknown_classes": unknown,
        "evaluation_class_order": eval_order,
        "datasets": {},
    }

    for mode in ("all_class", "unknown_class", "known_class"):
        if mode == "known_class":
            mode_order = list(trained_classes or known)
        else:
            mode_order = unknown if mode == "unknown_class" and unknown_class_prompt_only else eval_order
        source_id_to_mode_id = {idx: mode_order.index(name) for idx, name in source_names_by_id.items() if name in mode_order}
        mode_id_to_name = {idx: name for name, idx in ((name, mode_order.index(name)) for name in mode_order)}
        mode_root = output_root / mode
        image_out = mode_root / "images" / split
        label_out = mode_root / "labels" / split
        labels_root = mode_root / "labels"
        if label_out.exists():
            shutil.rmtree(label_out)
        if labels_root.exists():
            for cache_path in labels_root.glob("*.cache"):
                cache_path.unlink()
        label_out.mkdir(parents=True, exist_ok=True)
        image_out.mkdir(parents=True, exist_ok=True)

        annotation_count = 0
        class_annotation_counts = {name: 0 for name in mode_order}
        image_count = 0
        for image in images:
            rel = image.relative_to(image_root)
            link_or_copy(image, image_out / rel)
            source_label = label_path_for_image(image, image_root, label_root)
            target_label = label_out / rel.with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            if source_label.exists():
                for raw_line in source_label.read_text(encoding="utf-8").splitlines():
                    parts = raw_line.split()
                    if len(parts) < 5:
                        continue
                    source_id = int(float(parts[0]))
                    if source_id not in source_id_to_mode_id:
                        continue
                    if mode == "unknown_class" and source_id not in unknown_source_ids:
                        continue
                    mode_id = source_id_to_mode_id[source_id]
                    lines.append(" ".join([str(mode_id), *parts[1:5]]))
                    class_annotation_counts[mode_id_to_name[mode_id]] += 1
            target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            annotation_count += len(lines)
            image_count += 1

        yaml_path = mode_root / f"{mode}.yaml"
        yaml_config = {
            "path": str(mode_root.resolve()),
            "train": f"images/{split}",
            "val": f"images/{split}",
            "names": {idx: name for idx, name in enumerate(mode_order)},
        }
        yaml_path.write_text(yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8")
        dataset_info["datasets"][mode] = {
            "yaml": str(yaml_path),
            "image_count": image_count,
            "annotation_count": annotation_count,
            "class_annotation_counts": class_annotation_counts,
            "class_order": mode_order,
        }
        if logger:
            logger.info(
                "Research eval dataset prepared: mode=%s yaml=%s images=%s annotations=%s",
                mode,
                yaml_path,
                image_count,
                annotation_count,
            )

    save_json(output_root / "research_eval_datasets.json", dataset_info)
    return dataset_info


def to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    for attr in ("results_dict", "save_dir"):
        if hasattr(obj, attr):
            try:
                return to_serializable(getattr(obj, attr))
            except Exception:
                pass
    return str(obj)


def metrics_results_dict(metrics: Any) -> dict[str, Any]:
    if isinstance(metrics, dict):
        if "results_dict" in metrics and isinstance(metrics["results_dict"], dict):
            return metrics["results_dict"]
        return metrics
    results = getattr(metrics, "results_dict", None)
    if isinstance(results, dict):
        return results
    serial = to_serializable(metrics)
    if isinstance(serial, dict):
        return metrics_results_dict(serial)
    return {}


def metric_float(data: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in data:
            try:
                return float(data[key])
            except (TypeError, ValueError):
                return None
    return None


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def summarize_metrics(metrics: Any) -> dict[str, Any]:
    data = metrics_results_dict(metrics)
    precision = metric_float(data, "metrics/precision(B)", "precision", "P")
    recall = metric_float(data, "metrics/recall(B)", "recall", "R")
    map50 = metric_float(data, "metrics/mAP50(B)", "mAP50", "map50")
    map50_95 = metric_float(data, "metrics/mAP50-95(B)", "mAP50-95", "map50_95")
    names = normalize_metric_names(getattr(metrics, "names", None))
    per_class = extract_per_class_metrics(metrics, names)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mAP50": map50,
        "mAP50-95": map50_95,
        "mAP": map50_95,
        "mIoU": None,
        "speed": to_serializable(getattr(metrics, "speed", {})),
        "per_class": per_class,
        "raw_results": data,
    }


def normalize_metric_names(names: Any) -> dict[int, str]:
    try:
        return normalize_names(names)
    except Exception:
        return {}


def list_float_attr(obj: Any, attr: str) -> list[float]:
    value = getattr(obj, attr, None)
    if value is None:
        return []
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [float(item) for item in value]
    except Exception:
        return []


def extract_per_class_metrics(metrics: Any, names: dict[int, str]) -> dict[str, dict[str, float | None]]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    class_ids = list_float_attr(box, "ap_class_index")
    class_ids = [int(class_id) for class_id in class_ids]
    values = {
        "precision": list_float_attr(box, "p"),
        "recall": list_float_attr(box, "r"),
        "mAP50": list_float_attr(box, "ap50"),
        "mAP50-95": list_float_attr(box, "ap"),
        "mAP": list_float_attr(box, "ap"),
    }
    rows: dict[str, dict[str, float | None]] = {}
    for class_id, name in names.items():
        rows[name] = {metric: None for metric in values}
        rows[name]["f1"] = None

    metric_count = max([len(items) for items in values.values()] + [len(class_ids)])
    for idx in range(metric_count):
        class_id = class_ids[idx] if idx < len(class_ids) else idx
        name = names.get(class_id, str(class_id))
        rows.setdefault(name, {metric: None for metric in values})
        for metric, items in values.items():
            if idx < len(items):
                rows[name][metric] = items[idx]
        rows[name]["f1"] = f1_score(rows[name].get("precision"), rows[name].get("recall"))
    return rows


def model_parameter_count(model: Any) -> int | None:
    try:
        module = getattr(model, "model", model)
        return int(sum(parameter.numel() for parameter in module.parameters()))
    except Exception:
        return None


def xywhn_to_xyxy(values: list[float]) -> list[float]:
    x, y, w, h = values[:4]
    return [x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def read_ground_truth_boxes(label_path: Path) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xyxy = xywhn_to_xyxy([float(value) for value in parts[1:5]])
        except ValueError:
            continue
        boxes.append({"cls": cls, "box": xyxy})
    return boxes


def prediction_boxes_from_result(result: Any) -> list[dict[str, Any]]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None:
        return []
    cls_values = getattr(boxes_obj, "cls", None)
    xyxyn_values = getattr(boxes_obj, "xyxyn", None)
    if cls_values is None or xyxyn_values is None:
        return []
    try:
        if hasattr(cls_values, "detach"):
            cls_values = cls_values.detach().cpu()
        if hasattr(xyxyn_values, "detach"):
            xyxyn_values = xyxyn_values.detach().cpu()
        cls_list = cls_values.tolist()
        xyxyn_list = xyxyn_values.tolist()
    except Exception:
        return []
    boxes = []
    for cls, box in zip(cls_list, xyxyn_list):
        boxes.append({"cls": int(cls), "box": [float(value) for value in box[:4]]})
    return boxes


def mean_iou_for_model(
    model: Any,
    data_yaml: Path | str,
    class_order: list[str],
    split: str = "val",
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0",
    batch: int = 8,
    workers: int = 8,
    project: Path | None = None,
    name: str = "miou",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    config = load_yaml(Path(data_yaml))
    image_root = resolve_split_path(Path(data_yaml), config, split)
    label_root = label_root_from_image_root(image_root)
    images = find_images(image_root)
    if not images:
        return {"mIoU": None, "per_class": {}, "matches": 0}

    per_class_ious: dict[str, list[float]] = {name: [] for name in class_order}
    matched_ious: list[float] = []
    try:
        results = model.predict(
            source=str(image_root),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            workers=workers,
            project=str(project) if project else None,
            name=name,
            exist_ok=True,
            verbose=False,
            stream=True,
        )
        result_by_path = {Path(getattr(result, "path", "")).resolve(): result for result in results}
    except Exception as exc:
        if logger:
            logger.warning("mIoU prediction pass failed for %s: %s", data_yaml, exc)
        return {"mIoU": None, "per_class": {}, "matches": 0, "error": str(exc)}

    for image in images:
        result = result_by_path.get(image.resolve())
        if result is None:
            continue
        gt_boxes = read_ground_truth_boxes(label_path_for_image(image, image_root, label_root))
        pred_boxes = prediction_boxes_from_result(result)
        used_pred: set[int] = set()
        for gt in gt_boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, pred in enumerate(pred_boxes):
                if idx in used_pred or pred["cls"] != gt["cls"]:
                    continue
                candidate_iou = box_iou(gt["box"], pred["box"])
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_idx = idx
            if best_idx >= 0:
                used_pred.add(best_idx)
                matched_ious.append(best_iou)
                class_name = class_order[gt["cls"]] if 0 <= gt["cls"] < len(class_order) else str(gt["cls"])
                per_class_ious.setdefault(class_name, []).append(best_iou)

    per_class = {
        name: (sum(values) / len(values) if values else None)
        for name, values in per_class_ious.items()
    }
    return {
        "mIoU": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "per_class": per_class,
        "matches": len(matched_ious),
    }


def attach_miou(summary: dict[str, Any], miou_result: dict[str, Any]) -> dict[str, Any]:
    summary["mIoU"] = miou_result.get("mIoU")
    per_class_miou = miou_result.get("per_class") or {}
    for name, value in per_class_miou.items():
        summary.setdefault("per_class", {}).setdefault(name, {})["mIoU"] = value
    summary.setdefault("raw_results", {})["mIoU_result"] = miou_result
    return summary


def zero_class_metrics() -> dict[str, float]:
    return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0, "mAP": 0.0, "mIoU": 0.0, "f1": 0.0}


def none_class_metrics() -> dict[str, float | None]:
    return {"precision": None, "recall": None, "mAP50": None, "mAP50-95": None, "mAP": None, "mIoU": None, "f1": None}


def zero_summary_for_classes(class_order: list[str], class_counts: dict[str, int] | None = None, note: str = "") -> dict[str, Any]:
    per_class = {}
    counts = class_counts or {}
    for name in class_order:
        per_class[name] = zero_class_metrics() if counts.get(name, 0) > 0 else none_class_metrics()
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "mAP": 0.0,
        "mIoU": 0.0,
        "speed": {},
        "per_class": per_class,
        "raw_results": {"note": note} if note else {},
    }


def expand_known_summary_to_all_classes(
    known_summary: dict[str, Any],
    all_class_order: list[str],
    known_classes: list[str],
    class_counts: dict[str, int],
) -> dict[str, Any]:
    present_classes = [name for name in all_class_order if class_counts.get(name, 0) > 0]
    known_present = [name for name in known_classes if class_counts.get(name, 0) > 0]
    denominator = len(present_classes)
    scale = (len(known_present) / denominator) if denominator else 0.0

    precision = (known_summary.get("precision") or 0.0) * scale
    recall = (known_summary.get("recall") or 0.0) * scale
    expanded = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mAP50": (known_summary.get("mAP50") or 0.0) * scale,
        "mAP50-95": (known_summary.get("mAP50-95") or 0.0) * scale,
        "mAP": (known_summary.get("mAP") or known_summary.get("mAP50-95") or 0.0) * scale,
        "mIoU": (known_summary.get("mIoU") or 0.0) * scale,
        "speed": known_summary.get("speed", {}),
        "per_class": {},
        "raw_results": {
            **(known_summary.get("raw_results") or {}),
            "note": (
                "Standard YOLO known-class model has a 3-class closed-set head. "
                "Known classes were evaluated normally; untrained classes with GT were counted as zero "
                "in the all-class macro average."
            ),
            "macro_average_scale": scale,
            "present_classes": present_classes,
            "known_present_classes": known_present,
        },
    }

    known_per_class = known_summary.get("per_class") or {}
    for name in all_class_order:
        if name in known_classes:
            expanded["per_class"][name] = known_per_class.get(name, none_class_metrics())
        else:
            expanded["per_class"][name] = zero_class_metrics() if class_counts.get(name, 0) > 0 else none_class_metrics()
    return expanded


def merge_known_and_unknown_summary(
    known_summary: dict[str, Any],
    unknown_summary: dict[str, Any],
    all_class_order: list[str],
    known_classes: list[str],
    unknown_classes: list[str],
    class_counts: dict[str, int],
) -> dict[str, Any]:
    known_set = set(known_classes)
    unknown_set = set(unknown_classes)
    known_per_class = known_summary.get("per_class") or {}
    unknown_per_class = unknown_summary.get("per_class") or {}
    per_class: dict[str, dict[str, float | None]] = {}

    for name in all_class_order:
        if class_counts.get(name, 0) <= 0:
            per_class[name] = none_class_metrics()
        elif name in known_set:
            per_class[name] = known_per_class.get(name, zero_class_metrics())
        elif name in unknown_set:
            per_class[name] = unknown_per_class.get(name, zero_class_metrics())
        else:
            per_class[name] = zero_class_metrics()

    present_rows = [row for name, row in per_class.items() if class_counts.get(name, 0) > 0]

    def average(metric: str) -> float | None:
        values = [row.get(metric) for row in present_rows if row.get(metric) is not None]
        if not values:
            return None
        return sum(float(value) for value in values) / len(values)

    precision = average("precision")
    recall = average("recall")
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "mAP50": average("mAP50"),
        "mAP50-95": average("mAP50-95"),
        "mAP": average("mAP"),
        "mIoU": average("mIoU"),
        "speed": known_summary.get("speed", {}),
        "per_class": per_class,
        "raw_results": {
            **(known_summary.get("raw_results") or {}),
            "note": (
                "YOLO-World known-class all-class summary is synthesized from the fine-tuned "
                "known-class checkpoint for known classes and the unknown-class evaluation branch "
                "for unknown classes, so per-class unknown metrics match the unknown-only table."
            ),
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "present_classes": [name for name in all_class_order if class_counts.get(name, 0) > 0],
        },
    }


def save_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_type",
        "training_config",
        "scale",
        "run_dir",
        "checkpoint",
        "all_mAP50",
        "all_mAP50_95",
        "all_mAP",
        "all_mIoU",
        "all_precision",
        "all_recall",
        "all_f1",
        "unknown_mAP50",
        "unknown_mAP50_95",
        "unknown_mAP",
        "unknown_mIoU",
        "unknown_precision",
        "unknown_recall",
        "unknown_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def set_yoloworld_classes(model: Any, classes: list[str], logger: logging.Logger | None = None) -> None:
    if not hasattr(model, "set_classes"):
        return
    try:
        model.set_classes(classes)
    except RuntimeError as exc:
        if "Expected all tensors to be on the same device" not in str(exc):
            raise
        if logger:
            logger.warning("YOLO-World set_classes hit device mismatch. Retrying on CPU.")
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to("cpu")
        model.set_classes(classes)
    if logger:
        logger.info("YOLO-World evaluation classes set: %s", classes)


def evaluate_research_metrics(
    model: Any,
    model_type: str,
    source_data_yaml: Path,
    output_dir: Path,
    trained_classes: list[str],
    known_classes: list[str] | None = None,
    unknown_classes: list[str] | None = None,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0",
    workers: int = 8,
    run_dir: Path | None = None,
    checkpoint: Path | None = None,
    scale: str = "",
    training_config: str = "",
    zero_shot_unknown_model: str = "",
    use_zero_shot_unknown_model: bool = False,
    unknown_conf: float | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info = build_research_eval_datasets(
        source_data_yaml=source_data_yaml,
        output_root=output_dir / "datasets",
        trained_classes=trained_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
        split=split,
        unknown_class_prompt_only=(model_type == "yoloworld"),
        logger=logger,
    )
    eval_order = list(dataset_info["evaluation_class_order"])
    metrics_by_mode: dict[str, Any] = {}

    eval_training_config = training_config or infer_training_config(trained_classes, eval_order)

    for mode in ("all_class", "unknown_class"):
        mode_info = dataset_info["datasets"][mode]
        if model_type == "yolo" and eval_training_config == "known_class":
            if mode == "all_class":
                known_info = dataset_info["datasets"]["known_class"]
                if logger:
                    logger.info(
                        "Standard YOLO known-class all_class fallback: evaluating known classes on %s, then macro-expanding to all classes.",
                        known_info["yaml"],
                    )
                known_metrics = model.val(
                    data=known_info["yaml"],
                    split="val",
                    imgsz=imgsz,
                    batch=batch,
                    conf=conf,
                    iou=iou,
                    device=device,
                    workers=workers,
                    project=str(output_dir),
                    name="known_class_for_all",
                    exist_ok=True,
                )
                known_summary = summarize_metrics(known_metrics)
                known_miou = mean_iou_for_model(
                    model=model,
                    data_yaml=known_info["yaml"],
                    class_order=list(known_info.get("class_order") or trained_classes or known_classes or DEFAULT_KNOWN_CLASSES),
                    split="val",
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    batch=batch,
                    workers=workers,
                    project=output_dir,
                    name="known_class_for_all_miou",
                    logger=logger,
                )
                known_summary = attach_miou(known_summary, known_miou)
                summary = expand_known_summary_to_all_classes(
                    known_summary=known_summary,
                    all_class_order=list(mode_info.get("class_order") or eval_order),
                    known_classes=list(trained_classes or known_classes or DEFAULT_KNOWN_CLASSES),
                    class_counts=mode_info.get("class_annotation_counts") or {},
                )
            else:
                summary = zero_summary_for_classes(
                    class_order=list(mode_info.get("class_order") or []),
                    class_counts=mode_info.get("class_annotation_counts") or {},
                    note="Standard YOLO known-class is closed-set; unknown-class labels are outside the trained classifier head.",
                )
            metrics_by_mode[mode] = {
                "source_model": str(checkpoint) if checkpoint else "",
                "conf": conf,
                "parameters": model_parameter_count(model),
                "raw": summary["raw_results"],
                "summary": summary,
            }
            save_json(output_dir / f"{mode}_metrics.json", metrics_by_mode[mode])
            if logger:
                logger.info("Research evaluation completed with Standard YOLO known-class fallback for mode=%s summary=%s", mode, summary)
            continue

        eval_model = model
        eval_conf = conf
        source_model = str(checkpoint) if checkpoint else ""
        if (
            mode == "unknown_class"
            and model_type == "yoloworld"
            and eval_training_config == "known_class"
            and use_zero_shot_unknown_model
            and zero_shot_unknown_model
        ):
            if logger:
                logger.info("Using separate YOLO-World zero-shot unknown model for unknown_class eval: %s", zero_shot_unknown_model)
            eval_model = load_model(Path(zero_shot_unknown_model), "yoloworld")
            eval_conf = unknown_conf if unknown_conf is not None else conf
            source_model = zero_shot_unknown_model

        if model_type == "yoloworld":
            set_yoloworld_classes(eval_model, list(mode_info.get("class_order") or eval_order), logger)
        if logger:
            logger.info("Research evaluation start: mode=%s data=%s conf=%s source_model=%s", mode, mode_info["yaml"], eval_conf, source_model)
        metrics = eval_model.val(
            data=mode_info["yaml"],
            split="val",
            imgsz=imgsz,
            batch=batch,
            conf=eval_conf,
            iou=iou,
            device=device,
            workers=workers,
            project=str(output_dir),
            name=mode,
            exist_ok=True,
        )
        summary = summarize_metrics(metrics)
        miou_result = mean_iou_for_model(
            model=eval_model,
            data_yaml=mode_info["yaml"],
            class_order=list(mode_info.get("class_order") or eval_order),
            split="val",
            imgsz=imgsz,
            conf=eval_conf,
            iou=iou,
            device=device,
            batch=batch,
            workers=workers,
            project=output_dir,
            name=f"{mode}_miou",
            logger=logger,
        )
        summary = attach_miou(summary, miou_result)
        metrics_by_mode[mode] = {
            "source_model": source_model,
            "conf": eval_conf,
            "parameters": model_parameter_count(eval_model),
            "raw": to_serializable(metrics),
            "summary": summary,
        }
        save_json(output_dir / f"{mode}_metrics.json", metrics_by_mode[mode])
        if logger:
            logger.info("Research evaluation finished: mode=%s summary=%s", mode, metrics_by_mode[mode]["summary"])

    if model_type == "yoloworld" and eval_training_config == "known_class":
        all_info = dataset_info["datasets"].get("all_class") or {}
        all_metrics = metrics_by_mode.get("all_class") or {}
        unknown_metrics = metrics_by_mode.get("unknown_class") or {}
        if all_metrics.get("summary") and unknown_metrics.get("summary"):
            merged_summary = merge_known_and_unknown_summary(
                known_summary=all_metrics["summary"],
                unknown_summary=unknown_metrics["summary"],
                all_class_order=list(all_info.get("class_order") or eval_order),
                known_classes=list(known_classes or DEFAULT_KNOWN_CLASSES),
                unknown_classes=list(unknown_classes or DEFAULT_UNKNOWN_CLASSES),
                class_counts=all_info.get("class_annotation_counts") or {},
            )
            all_metrics["summary"] = merged_summary
            all_metrics["raw"] = merged_summary["raw_results"]
            all_metrics["merged_sources"] = {
                "known": {
                    "source_model": all_metrics.get("source_model"),
                    "conf": all_metrics.get("conf"),
                },
                "unknown": {
                    "source_model": unknown_metrics.get("source_model"),
                    "conf": unknown_metrics.get("conf"),
                },
            }
            metrics_by_mode["all_class"] = all_metrics
            save_json(output_dir / "all_class_metrics.json", all_metrics)
            if logger:
                logger.info("YOLO-World known-class all_class summary merged from known and unknown branches: %s", merged_summary)

    all_summary = metrics_by_mode.get("all_class", {}).get("summary", {})
    unknown_summary = metrics_by_mode.get("unknown_class", {}).get("summary", {})
    row = {
        "model_type": model_type,
        "training_config": eval_training_config,
        "scale": scale,
        "run_dir": str(run_dir) if run_dir else "",
        "checkpoint": str(checkpoint) if checkpoint else "",
        "all_mAP50": all_summary.get("mAP50"),
        "all_mAP50_95": all_summary.get("mAP50-95"),
        "all_mAP": all_summary.get("mAP") if all_summary.get("mAP") is not None else all_summary.get("mAP50-95"),
        "all_mIoU": all_summary.get("mIoU"),
        "all_precision": all_summary.get("precision"),
        "all_recall": all_summary.get("recall"),
        "all_f1": all_summary.get("f1"),
        "unknown_mAP50": unknown_summary.get("mAP50"),
        "unknown_mAP50_95": unknown_summary.get("mAP50-95"),
        "unknown_mAP": unknown_summary.get("mAP") if unknown_summary.get("mAP") is not None else unknown_summary.get("mAP50-95"),
        "unknown_mIoU": unknown_summary.get("mIoU"),
        "unknown_precision": unknown_summary.get("precision"),
        "unknown_recall": unknown_summary.get("recall"),
        "unknown_f1": unknown_summary.get("f1"),
    }
    result = {"datasets": dataset_info, "metrics": metrics_by_mode, "summary_row": row}
    save_json(output_dir / "research_metrics_summary.json", result)
    save_summary_csv(output_dir / "research_summary.csv", [row])
    return result


def infer_training_config(trained_classes: list[str], eval_order: list[str]) -> str:
    if trained_classes and len(trained_classes) == len(eval_order) and set(trained_classes) == set(eval_order):
        return "all_class"
    return "known_class"


def read_run_args(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "configs" / "args.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "args" in data and isinstance(data["args"], dict):
        return data["args"]
    return data if isinstance(data, dict) else {}


def read_trained_classes(run_dir: Path) -> list[str]:
    config_path = run_dir / "configs" / "config_used.yaml"
    if not config_path.exists():
        return []
    config = load_yaml(config_path)
    return [name for _, name in sorted(normalize_names(config.get("names", {})).items())]


def infer_model_type(run_dir: Path, requested: str = "auto") -> str:
    if requested in {"yolo", "yoloworld"}:
        return requested
    text = str(run_dir).lower()
    if "yoloworld" in text or "world" in text:
        return "yoloworld"
    return "yolo"


def infer_scale(run_dir: Path, checkpoint: Path | None = None, args: dict[str, Any] | None = None) -> str:
    text = f"{run_dir.name} {checkpoint or ''} {(args or {}).get('model', '')}".lower()
    for scale in ("x", "l", "m", "s", "n"):
        if f"_{scale}" in text or f"-{scale}" in text or f"yolov8{scale}" in text:
            return scale
    return ""


def run_dir_checkpoint(run_dir: Path, checkpoint_name: str = "best.pt") -> Path:
    checkpoint = run_dir / "weights" / checkpoint_name
    if checkpoint.exists():
        return checkpoint
    fallback = run_dir / "weights" / "last.pt"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {run_dir / 'weights'}")


def load_model(checkpoint: Path, model_type: str) -> Any:
    if model_type == "yoloworld":
        from ultralytics import YOLOWorld

        return YOLOWorld(str(checkpoint))
    from ultralytics import YOLO

    return YOLO(str(checkpoint))


def evaluate_run_dir(
    run_dir: Path,
    source_data_yaml: Path,
    output_name: str = "research_eval",
    model_type: str = "auto",
    known_classes: list[str] | None = None,
    unknown_classes: list[str] | None = None,
    checkpoint_name: str = "best.pt",
    split: str = "val",
    imgsz: int = 640,
    batch: int = 8,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = "0",
    workers: int = 8,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    checkpoint = run_dir_checkpoint(run_dir, checkpoint_name)
    args = read_run_args(run_dir)
    detected_type = infer_model_type(run_dir, model_type)
    trained_classes = read_trained_classes(run_dir)
    if not trained_classes:
        if detected_type == "yoloworld":
            trained_classes = parse_csv(str(args.get("known_classes", "")))
        else:
            trained_classes = parse_csv(str(args.get("train_classes", "")))
    if not trained_classes:
        source_names = [name for _, name in sorted(normalize_names(load_yaml(source_data_yaml).get("names", {})).items())]
        trained_classes = source_names
    source_names = [name for _, name in sorted(normalize_names(load_yaml(source_data_yaml).get("names", {})).items())]
    known = known_classes or DEFAULT_KNOWN_CLASSES
    unknown = infer_unknown_classes(source_names, known, unknown_classes)
    eval_order = class_order_for_evaluation(source_names, trained_classes, known, unknown)
    training_config = infer_training_config(trained_classes, eval_order)
    scale = infer_scale(run_dir, checkpoint, args)
    if logger:
        logger.info(
            "Evaluating run: run_dir=%s model_type=%s training_config=%s scale=%s checkpoint=%s",
            run_dir,
            detected_type,
            training_config,
            scale,
            checkpoint,
        )
    model = load_model(checkpoint, detected_type)
    return evaluate_research_metrics(
        model=model,
        model_type=detected_type,
        source_data_yaml=source_data_yaml,
        output_dir=run_dir / output_name,
        trained_classes=trained_classes,
        known_classes=known,
        unknown_classes=unknown,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        workers=workers,
        run_dir=run_dir,
        checkpoint=checkpoint,
        scale=scale,
        training_config=training_config,
        zero_shot_unknown_model=str(args.get("zero_shot_unknown_model", "")),
        use_zero_shot_unknown_model=bool(args.get("use_zero_shot_unknown_model", False)),
        unknown_conf=float(args["unknown_conf_thres"]) if "unknown_conf_thres" in args else None,
        logger=logger,
    )


def discover_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if (root / "weights").exists() else sorted(item for item in root.iterdir() if item.is_dir())
        for candidate in candidates:
            if (candidate / "weights").exists() and ((candidate / "weights" / "best.pt").exists() or (candidate / "weights" / "last.pt").exists()):
                run_dirs.append(candidate)
    return sorted(dict.fromkeys(run_dirs))


def configure_file_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"research_evaluation.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
