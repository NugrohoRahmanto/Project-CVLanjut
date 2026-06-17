from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


BDD10K_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}

DEFAULT_KNOWN_CLASSES = "car,bus,truck"
DEFAULT_UNKNOWN_PROMPTS = "pedestrian,rider,train,motorcycle,bicycle,traffic light,traffic sign"
UNKNOWN_OBJECT_LABEL = "Unknown Object"

OFFICIAL_ULTRALYTICS_WORLD_MODELS = {
    "yolov8s-world.pt",
    "yolov8m-world.pt",
    "yolov8l-world.pt",
    "yolov8x-world.pt",
    "yolov8s-worldv2.pt",
    "yolov8m-worldv2.pt",
    "yolov8l-worldv2.pt",
    "yolov8x-worldv2.pt",
}


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def timestamped_name(name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    parser.add_argument("--model", default="yolov8s-world.pt")
    parser.add_argument("--output-dir", default="runs/yoloworld_bdd10k")
    parser.add_argument("--experiment-name", default="yoloworld_bdd10k_finetune")
    parser.add_argument("--timestamp-output", action="store_true", help="Prefix experiment name with yyyymmdd_hhmmss.")
    parser.add_argument("--known-classes", default=DEFAULT_KNOWN_CLASSES, help="Comma-separated BDD10K classes used for supervised training.")
    parser.add_argument("--unknown-classes", default="", help="Optional comma-separated unknown classes. Default: all BDD10K classes not in known classes.")
    parser.add_argument("--unknown-prompts", default=DEFAULT_UNKNOWN_PROMPTS, help="Comma-separated zero-shot prompts appended for open-vocabulary predict/eval.")
    parser.add_argument("--zero-shot-unknown-model", default="yolov8s-world.pt", help="Pretrained YOLO-World weight used only for unknown zero-shot sample/predict post-processing.")
    parser.add_argument("--use-zero-shot-unknown-model", action=argparse.BooleanOptionalAction, default=True, help="Use a separate pretrained YOLO-World model for unknown prompts.")
    parser.add_argument("--unknown-conf-thres", type=float, default=0.05, help="Confidence threshold for zero-shot unknown prompt detection.")
    parser.add_argument("--freeze-text-encoder", action=argparse.BooleanOptionalAction, default=True, help="Freeze YOLO-World text/CLIP encoder modules before fine-tuning.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", default=False, nargs="?", const=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze", type=int, nargs="*", default=None)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-split", choices=("train", "val"), default="val", help="Dataset split used by --eval-only. Default: val.")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--source", default="")
    parser.add_argument("--save-eval-samples", action=argparse.BooleanOptionalAction, default=True, help="Save sample inference visualizations after train/eval.")
    parser.add_argument("--sample-source", default="", help="Image/folder source for sample inference visualizations. Default: BDD10K val images.")
    parser.add_argument("--sample-count", type=int, default=16, help="Number of sample images to visualize.")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-format", default="onnx")
    parser.add_argument("--skip-filtered-dataset", action="store_true", help="Use data yaml as-is without filtering known classes.")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict YOLO-World on BDD10K with Ultralytics.")
    return add_common_args(parser)


def setup_logger(experiment_dir: Path) -> logging.Logger:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    root_log = Path("training.log")
    exp_log = experiment_dir / "logs" / "train.log"
    exp_log.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s/%(funcName)s | %(message)s")
    logger = logging.getLogger("yoloworld_bdd10k")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    for handler in (logging.FileHandler(root_log, mode="w"), logging.FileHandler(exp_log, mode="a")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


@contextlib.contextmanager
def redirect_console_to_file(experiment_dir: Path):
    console_log = experiment_dir / "logs" / "console.log"
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("a", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            yield console_log


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError("data yaml must define names as a dict or list")


def resolve_split_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def collect_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch

        versions.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda})
    except Exception as exc:
        versions["torch_error"] = str(exc)
    for module_name in ("ultralytics", "numpy"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[f"{module_name}_error"] = str(exc)
    return versions


def validate_device(device: str, logger: logging.Logger) -> None:
    if device in ("", "cpu", "mps"):
        return
    try:
        import torch

        wants_cuda = any(part.strip().isdigit() for part in str(device).split(","))
        if wants_cuda and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device '{device}' was requested, but torch.cuda.is_available() is False.")
    except ImportError as exc:
        raise RuntimeError("PyTorch is required before selecting a CUDA device.") from exc
    logger.info("Device check passed for device=%s", device)


def ensure_model_reference(model: str) -> None:
    if Path(model).exists():
        return
    if model in OFFICIAL_ULTRALYTICS_WORLD_MODELS:
        return
    if model.startswith(("yolo", "http://", "https://")):
        return
    raise FileNotFoundError(
        f"Model '{model}' does not exist locally and does not look like an official Ultralytics model name. "
        "Use e.g. yolov8s-world.pt, yolov8s-worldv2.pt, or a valid checkpoint path."
    )


def find_images(split_dir: Path) -> list[Path]:
    if not split_dir or not split_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in exts)


def label_path_for_image(image: Path, image_root: Path, label_root: Path) -> Path:
    return label_root / image.relative_to(image_root).with_suffix(".txt")


def bbox_to_yolo(box: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(box["x1"]), float(width)))
    y1 = max(0.0, min(float(box["y1"]), float(height)))
    x2 = max(0.0, min(float(box["x2"]), float(width)))
    y2 = max(0.0, min(float(box["y2"]), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, (x2 - x1) / width, (y2 - y1) / height)


def image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to auto-convert BDD10K JSON annotations to YOLO labels.") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image while converting labels: {path}")
    height, width = image.shape[:2]
    return width, height


def label_dir_has_files(label_dir: Path) -> bool:
    return count_yolo_annotations(label_dir) > 0


def auto_convert_bdd_json_split(dataset_root: Path, image_dir: Path, label_dir: Path, split: str, names: dict[int, str], logger: logging.Logger) -> int:
    if label_dir_has_files(label_dir):
        return 0
    json_path = dataset_root / "labels" / f"bdd100k_labels_images_{split}.json"
    if not json_path.exists():
        logger.warning("No YOLO labels and no BDD JSON for split=%s at %s", split, json_path)
        return 0
    images = find_images(image_dir)
    if not images:
        logger.warning("No images found for split=%s at %s", split, image_dir)
        return 0
    image_by_name = {image.name: image for image in images}
    class_to_id = {name: idx for idx, name in names.items()}
    logger.info("Auto-converting BDD JSON to YOLO labels for split=%s from %s", split, json_path)
    records = json.loads(json_path.read_text(encoding="utf-8"))
    matched = 0
    annotations = 0
    label_dir.mkdir(parents=True, exist_ok=True)
    for item in records:
        image = image_by_name.get(item.get("name", ""))
        if image is None:
            continue
        width, height = image_size(image)
        lines: list[str] = []
        for label in item.get("labels", []):
            category = label.get("category")
            box = label.get("box2d")
            if category not in class_to_id or not box:
                continue
            converted = bbox_to_yolo(box, width, height)
            if converted is None:
                continue
            lines.append(f"{class_to_id[category]} " + " ".join(f"{value:.6f}" for value in converted))
        target = label_path_for_image(image, image_dir, label_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        matched += 1
        annotations += len(lines)
    logger.info("Auto-converted split=%s matched_images=%s annotations=%s output=%s", split, matched, annotations, label_dir)
    if matched == 0:
        logger.warning("BDD JSON split=%s did not match images in %s", split, image_dir)
    return annotations


def count_yolo_annotations(label_dir: Path) -> int:
    if not label_dir.exists():
        return 0
    total = 0
    for label_file in label_dir.rglob("*.txt"):
        total += sum(1 for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def prepare_known_dataset(args: argparse.Namespace, experiment_dir: Path, logger: logging.Logger) -> Path:
    data_yaml = Path(args.data_yaml)
    config = load_yaml(data_yaml)
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve() if not Path(config.get("path", "")).is_absolute() else dataset_root
        if not dataset_root.exists():
            dataset_root = Path(config.get("path", data_yaml.parent)).resolve()
    names = normalize_names(config.get("names", BDD10K_NAMES))
    name_to_id = {name: idx for idx, name in names.items()}
    known = parse_csv(args.known_classes)
    if not known:
        raise ValueError("--known-classes must contain at least one class name")
    missing = [name for name in known if name not in name_to_id]
    if missing:
        raise ValueError(f"Known classes not present in yaml names: {missing}")
    known_old_ids = {name_to_id[name]: new_id for new_id, name in enumerate(known)}
    unknown = parse_csv(args.unknown_classes) or [name for name in names.values() if name not in known]

    if args.skip_filtered_dataset:
        logger.info("Using data yaml as-is. Known classes=%s Unknown classes=%s", known, unknown)
        return data_yaml

    out_root = experiment_dir / "dataset_known"
    yaml_out = experiment_dir / "configs" / "config_used.yaml"
    out_root.mkdir(parents=True, exist_ok=True)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        image_dir = resolve_split_path(dataset_root, config.get(split))
        if image_dir is None:
            continue
        label_dir = Path(str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        auto_convert_bdd_json_split(dataset_root, image_dir, label_dir, split, names, logger)
        out_img_dir = out_root / "images" / split
        out_label_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for image in find_images(image_dir):
            target_image = out_img_dir / image.relative_to(image_dir)
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                try:
                    target_image.symlink_to(image.resolve())
                except OSError:
                    shutil.copy2(image, target_image)
            source_label = label_path_for_image(image, image_dir, label_dir)
            target_label = out_label_dir / image.relative_to(image_dir).with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            kept: list[str] = []
            if source_label.exists():
                for line in source_label.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    old_id = int(float(parts[0]))
                    if old_id in known_old_ids:
                        kept.append(" ".join([str(known_old_ids[old_id]), *parts[1:5]]))
            target_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    train_annotation_count = count_yolo_annotations(out_root / "labels" / "train")
    val_annotation_count = count_yolo_annotations(out_root / "labels" / "val")
    if train_annotation_count == 0:
        raise RuntimeError(
            "No known-class training annotations were found after filtering. "
            "Check that data/bdd10k/labels/train/*.txt exists or that bdd100k_labels_images_train.json matches data/bdd10k/images/train."
        )
    if val_annotation_count == 0:
        raise RuntimeError(
            "No known-class validation annotations were found after filtering images/val. "
            "Validation data must not be empty. Check data/bdd10k/labels/val/*.txt and --known-classes."
        )
    filtered_config = {"path": str(out_root.resolve()), "train": "images/train", "val": "images/val", "names": {i: n for i, n in enumerate(known)}}
    yaml_out.write_text(yaml.safe_dump(filtered_config, sort_keys=False), encoding="utf-8")
    logger.info("Created known-class dataset: %s", yaml_out)
    logger.info("Known classes: %s", known)
    logger.info("Unknown classes ignored during supervised training: %s", unknown)
    logger.info("Known-class train annotations: %s", train_annotation_count)
    logger.info("Known-class val annotations: %s", val_annotation_count)
    return yaml_out


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def to_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    for attr in ("results_dict", "save_dir"):
        if hasattr(obj, attr):
            try:
                return to_serializable(getattr(obj, attr))
            except Exception:
                pass
    return str(obj)


def copy_training_artifacts(save_dir: Path | None, experiment_dir: Path, logger: logging.Logger) -> Path:
    metrics_dir = experiment_dir / "metrics"
    configs_dir = experiment_dir / "configs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    if save_dir is None or not save_dir.exists() or save_dir.resolve() == experiment_dir.resolve():
        root_results = experiment_dir / "results.csv"
        if root_results.exists():
            target = metrics_dir / "training_history.csv"
            shutil.copy2(root_results, target)
            logger.info("Copied training history: %s -> %s", root_results, target)
            root_results.unlink()
        root_args = experiment_dir / "args.yaml"
        if root_args.exists():
            target = configs_dir / "ultralytics_args.yaml"
            shutil.copy2(root_args, target)
            logger.info("Copied Ultralytics args: %s -> %s", root_args, target)
            root_args.unlink()
        return experiment_dir
    logger.info("Ultralytics save_dir differs from experiment_dir: %s", save_dir)
    artifact_targets = {
        "results.csv": metrics_dir / "training_history.csv",
        "args.yaml": configs_dir / "ultralytics_args.yaml",
    }
    for name, target in artifact_targets.items():
        source = save_dir / name
        if source.exists():
            shutil.copy2(source, target)
            logger.info("Copied training artifact: %s -> %s", source, target)
    source_weights = save_dir / "weights"
    target_weights = experiment_dir / "weights"
    if source_weights.exists():
        shutil.copytree(source_weights, target_weights, dirs_exist_ok=True)
        logger.info("Copied weights directory: %s -> %s", source_weights, target_weights)
    return experiment_dir


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def log_epoch_metrics(experiment_dir: Path, logger: logging.Logger) -> dict[str, Any] | None:
    results_csv = experiment_dir / "metrics" / "training_history.csv"
    if not results_csv.exists():
        legacy_results = experiment_dir / "results.csv"
        if legacy_results.exists():
            results_csv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_results, results_csv)
    if not results_csv.exists():
        logger.warning("Epoch metrics results.csv not found at %s. Full Ultralytics epoch output is in console.log.", results_csv)
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        logger.warning("Epoch metrics results.csv is empty: %s", results_csv)
        return None
    logger.info("Epoch metrics from results.csv: %s", results_csv)
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        clean = {key.strip(): value for key, value in row.items()}
        cleaned_rows.append(clean)
        epoch = clean.get("epoch", "?")
        summary_keys = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "lr/pg0",
        ]
        summary = {key: clean[key] for key in summary_keys if key in clean}
        logger.info("Epoch %s metrics: %s", epoch, summary)
    final = cleaned_rows[-1]
    best_map_row = max(cleaned_rows, key=lambda row: parse_float(row.get("metrics/mAP50-95(B)")) or float("-inf"))
    summary = {
        "results_csv": str(results_csv),
        "num_epochs_logged": len(cleaned_rows),
        "final_epoch": final,
        "best_map50_95_epoch": best_map_row,
    }
    save_json(experiment_dir / "metrics" / "metrics_summary.json", summary)
    logger.info("Saved metrics summary: %s", experiment_dir / "metrics" / "metrics_summary.json")
    return summary


def save_metrics_csv(path: Path, metrics: Any, logger: logging.Logger) -> None:
    data = to_serializable(metrics)
    if isinstance(data, dict):
        rows = [{"metric": key, "value": value} for key, value in data.items()]
    else:
        rows = [{"metric": "result", "value": data}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved final metrics csv: %s", path)


def save_confidence_artifacts(rows: list[dict[str, Any]], experiment_dir: Path, logger: logging.Logger) -> None:
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for old_name in ("confidence_by_label.csv", "confidence_by_label.png"):
        old_path = metrics_dir / old_name
        if old_path.exists():
            old_path.unlink()
    csv_path = metrics_dir / "confidence_histogram.csv"
    chart_path = metrics_dir / "confidence_histogram.png"
    grouped: dict[str, list[float]] = {"known": [], "unknown": []}
    for row in rows:
        try:
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError):
            continue
        group = "unknown" if bool(row.get("is_unknown")) else "known"
        grouped[group].append(confidence)

    bins = [i / 10 for i in range(11)]
    histogram_rows: list[dict[str, Any]] = []
    for group, values in grouped.items():
        for start, end in zip(bins[:-1], bins[1:]):
            count = sum(1 for value in values if start <= value <= end) if end == 1.0 else sum(1 for value in values if start <= value < end)
            histogram_rows.append(
                {
                    "group": group,
                    "confidence_min": f"{start:.1f}",
                    "confidence_max": f"{end:.1f}",
                    "count": count,
                }
            )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "confidence_min", "confidence_max", "count"])
        writer.writeheader()
        writer.writerows(histogram_rows)
    logger.info("Saved confidence histogram csv: %s", csv_path)

    if not grouped["known"] and not grouped["unknown"]:
        logger.warning("No detections available for confidence chart.")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Could not import matplotlib for confidence chart: %s", exc)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if grouped["known"]:
        ax.hist(grouped["known"], bins=bins, alpha=0.7, label=f"Known (n={len(grouped['known'])})", color="#2ca02c", edgecolor="black")
    if grouped["unknown"]:
        ax.hist(grouped["unknown"], bins=bins, alpha=0.7, label=f"Unknown (n={len(grouped['unknown'])})", color="#d62728", edgecolor="black")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Detection count")
    ax.set_title("Confidence Histogram on Evaluation Samples")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    logger.info("Saved confidence histogram chart: %s", chart_path)


def default_sample_source(args: argparse.Namespace) -> Path:
    if args.sample_source:
        return Path(args.sample_source)
    val = Path("data/bdd10k/images/val")
    if val.exists():
        return val
    return Path("data/bdd10k/images/train")


def sample_image_paths(source: Path, count: int) -> list[Path]:
    if count <= 0:
        return []
    if source.is_file():
        return [source]
    return find_images(source)[:count]


def draw_labeled_box(image: Any, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    y_text = max(y1, text_h + baseline + 4)
    cv2.rectangle(image, (x1, y_text - text_h - baseline - 4), (x1 + text_w + 6, y_text + baseline), color, -1)
    cv2.putText(image, label, (x1 + 3, y_text - 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = values
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return [x1, y1, x2, y2]


def ground_truth_label_path(image_path: Path) -> Path:
    candidates: list[Path] = []
    text = str(image_path)
    if f"{os.sep}images{os.sep}" in text:
        candidates.append(Path(text.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt"))
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        split = parts[idx + 1] if idx + 1 < len(parts) else ""
        root = Path(*parts[:idx]) if idx > 0 else Path(".")
        if split:
            candidates.append(root / "labels" / split / image_path.with_suffix(".txt").name)
        for fallback_split in ("val", "train"):
            candidates.append(root / "labels" / fallback_split / image_path.with_suffix(".txt").name)
    candidates.append(image_path.with_suffix(".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def draw_ground_truth(image: Any, image_path: Path, names: dict[int, str], logger: logging.Logger) -> int:
    label_path = ground_truth_label_path(image_path)
    if not label_path.exists():
        logger.warning(
            "Ground-truth label not found for sample image: %s. "
            "Rerun: .venv/bin/python scripts/convert_bdd10k_to_yolo.py --data-root data/bdd10k and make sure val labels exist.",
            label_path,
        )
        return 0
    height, width = image.shape[:2]
    count = 0
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            logger.warning("Skipping invalid GT label row %s:%s", label_path, line_no)
            continue
        try:
            class_id = int(float(parts[0]))
            box = yolo_to_xyxy([float(value) for value in parts[1:]], width, height)
        except ValueError:
            logger.warning("Skipping non-numeric GT label row %s:%s", label_path, line_no)
            continue
        draw_labeled_box(image, box, names.get(class_id, str(class_id)), (255, 120, 0))
        count += 1
    return count


def add_panel_title(image: Any, title: str) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def save_sample_visualizations(model: Any, args: argparse.Namespace, experiment_dir: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    if not args.save_eval_samples:
        logger.info("Sample visualizations disabled.")
        return []
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python is not available; skipping sample visualizations.")
        return []

    source = default_sample_source(args)
    images = sample_image_paths(source, args.sample_count)
    if not images:
        logger.warning("No sample images found for visualization source=%s", source)
        return []

    evaluation_dir = experiment_dir / "evaluation"
    output_dir = evaluation_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    known = parse_csv(args.known_classes)
    unknown_prompts = parse_csv(args.unknown_prompts)
    set_world_classes(model, args, logger, include_unknown=False)
    logger.info("Sample known inference start: source=%s count=%s output=%s prompts=%s", source, len(images), output_dir, known)
    known_predictions = model.predict(
        source=[str(path) for path in images],
        conf=args.conf_thres,
        iou=args.iou_thres,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=False,
    )

    unknown_predictions = [None] * len(images)
    if args.use_zero_shot_unknown_model and unknown_prompts:
        from ultralytics import YOLOWorld

        ensure_model_reference(args.zero_shot_unknown_model)
        logger.info(
            "Sample zero-shot unknown inference start: model=%s prompts=%s conf=%s",
            args.zero_shot_unknown_model,
            unknown_prompts,
            args.unknown_conf_thres,
        )
        unknown_model = YOLOWorld(args.zero_shot_unknown_model)
        unknown_args = argparse.Namespace(**vars(args))
        unknown_args.known_classes = ",".join(unknown_prompts)
        unknown_args.unknown_prompts = ""
        set_world_classes(unknown_model, unknown_args, logger, include_unknown=False)
        unknown_predictions = unknown_model.predict(
            source=[str(path) for path in images],
            conf=args.unknown_conf_thres,
            iou=args.iou_thres,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )
    elif unknown_prompts:
        logger.info("Separate zero-shot unknown model disabled; falling back to same model with appended unknown prompts.")
        set_world_classes(model, args, logger, include_unknown=True)
        unknown_predictions = model.predict(
            source=[str(path) for path in images],
            conf=args.unknown_conf_thres,
            iou=args.iou_thres,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )

    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(known_predictions):
        image_path = images[idx] if idx < len(images) else Path(str(getattr(result, "path", "")))
        original = cv2.imread(str(image_path))
        if original is None:
            logger.warning("Could not read sample image for drawing: %s", image_path)
            continue
        gt_image = original.copy()
        pred_image = original.copy()
        gt_count = draw_ground_truth(gt_image, image_path, BDD10K_NAMES, logger)
        boxes = getattr(result, "boxes", None)
        image_rows: list[dict[str, Any]] = []
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for box, conf, class_id_value in zip(xyxy, confs, classes):
                class_id = int(class_id_value)
                prompt_label = known[class_id] if 0 <= class_id < len(known) else str(class_id)
                draw_labeled_box(pred_image, box, f"{prompt_label} {float(conf):.2f}", (0, 170, 0))
                item = {
                    "image": str(image_path),
                    "class_id": class_id,
                    "prompt_label": prompt_label,
                    "final_label": prompt_label,
                    "confidence": float(conf),
                    "xyxy": [float(v) for v in box],
                    "is_unknown": False,
                    "source_model": "fine_tuned_known",
                }
                image_rows.append(item)
                rows.append(item)
        unknown_result = unknown_predictions[idx] if idx < len(unknown_predictions) else None
        unknown_boxes = getattr(unknown_result, "boxes", None)
        if unknown_boxes is not None:
            xyxy = unknown_boxes.xyxy.cpu().tolist()
            confs = unknown_boxes.conf.cpu().tolist()
            classes = unknown_boxes.cls.cpu().tolist()
            for box, conf, class_id_value in zip(xyxy, confs, classes):
                class_id = int(class_id_value)
                prompt_label = unknown_prompts[class_id] if 0 <= class_id < len(unknown_prompts) else str(class_id)
                draw_labeled_box(pred_image, box, f"{UNKNOWN_OBJECT_LABEL} {float(conf):.2f}", (0, 0, 255))
                item = {
                    "image": str(image_path),
                    "class_id": len(known) + class_id,
                    "prompt_label": prompt_label,
                    "final_label": UNKNOWN_OBJECT_LABEL,
                    "confidence": float(conf),
                    "xyxy": [float(v) for v in box],
                    "is_unknown": True,
                    "source_model": "pretrained_zero_shot_unknown",
                }
                image_rows.append(item)
                rows.append(item)
        add_panel_title(gt_image, "Ground Truth")
        add_panel_title(pred_image, "Inference")
        combined = cv2.hconcat([gt_image, pred_image])
        output_image = output_dir / image_path.name
        cv2.imwrite(str(output_image), combined)
        logger.info(
            "Saved sample visualization: %s gt=%s detections=%s unknown=%s",
            output_image,
            gt_count,
            len(image_rows),
            sum(1 for row in image_rows if row["is_unknown"]),
        )
    save_json(evaluation_dir / "sample_predictions.json", rows)
    logger.info("Sample inference finished: images=%s detections=%s json=%s", len(images), len(rows), evaluation_dir / "sample_predictions.json")
    return rows


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def progress_interval(total: int) -> int:
    if total <= 0:
        return 1
    return max(1, total // 20)


def attach_progress_callbacks(model: Any, args: argparse.Namespace, logger: logging.Logger) -> None:
    state: dict[str, Any] = {
        "train_start": None,
        "epoch_start": None,
        "epoch_batch": 0,
        "train_batches": None,
        "val_start": None,
        "val_batch": 0,
        "val_batches": None,
        "predict_start": None,
        "predict_batch": 0,
        "predict_batches": None,
    }

    def on_pretrain_routine_start(trainer):
        logger.info("Stage: preparing model, optimizer, dataloaders, and training pipeline")

    def on_train_start(trainer):
        state["train_start"] = time.perf_counter()
        state["train_batches"] = len(getattr(trainer, "train_loader", []) or [])
        logger.info(
            "Training start: epochs=%s start_epoch=%s train_batches_per_epoch=%s batch_size=%s imgsz=%s device=%s",
            getattr(trainer, "epochs", args.epochs),
            getattr(trainer, "start_epoch", 0) + 1,
            state["train_batches"],
            getattr(trainer, "batch_size", args.batch_size),
            getattr(trainer.args, "imgsz", args.imgsz),
            getattr(trainer, "device", args.device),
        )

    def on_train_epoch_start(trainer):
        state["epoch_start"] = time.perf_counter()
        state["epoch_batch"] = 0
        total_epochs = getattr(trainer, "epochs", args.epochs)
        epoch = getattr(trainer, "epoch", 0) + 1
        logger.info("Training epoch start: %s/%s", epoch, total_epochs)

    def on_train_batch_end(trainer):
        total_batches = state.get("train_batches") or len(getattr(trainer, "train_loader", []) or [])
        state["epoch_batch"] = int(state.get("epoch_batch", 0)) + 1
        batch = state["epoch_batch"]
        interval = progress_interval(total_batches)
        if batch != 1 and batch != total_batches and batch % interval:
            return
        elapsed = time.perf_counter() - (state.get("epoch_start") or time.perf_counter())
        eta = None
        if batch > 0 and total_batches:
            eta = (elapsed / batch) * max(total_batches - batch, 0)
        epoch = getattr(trainer, "epoch", 0) + 1
        total_epochs = getattr(trainer, "epochs", args.epochs)
        losses = getattr(trainer, "tloss", None)
        loss_text = str(losses.detach().cpu().tolist()) if hasattr(losses, "detach") else str(losses)
        logger.info(
            "Training progress: epoch=%s/%s batch=%s/%s elapsed=%s eta=%s avg_loss=%s",
            epoch,
            total_epochs,
            batch,
            total_batches,
            format_seconds(elapsed),
            format_seconds(eta),
            loss_text,
        )

    def on_train_epoch_end(trainer):
        elapsed = time.perf_counter() - (state.get("epoch_start") or time.perf_counter())
        epoch = getattr(trainer, "epoch", 0) + 1
        total_epochs = getattr(trainer, "epochs", args.epochs)
        logger.info("Training epoch finished: %s/%s elapsed=%s", epoch, total_epochs, format_seconds(elapsed))

    def on_train_end(trainer):
        elapsed = time.perf_counter() - (state.get("train_start") or time.perf_counter())
        logger.info("Training finished: elapsed=%s save_dir=%s", format_seconds(elapsed), getattr(trainer, "save_dir", "unknown"))

    def on_val_start(validator):
        state["val_start"] = time.perf_counter()
        state["val_batch"] = 0
        dataloader = getattr(validator, "dataloader", None)
        state["val_batches"] = len(dataloader) if dataloader is not None else None
        logger.info("Evaluation/validation start: batches=%s", state["val_batches"])

    def on_val_batch_end(validator):
        total_batches = state.get("val_batches")
        state["val_batch"] = int(state.get("val_batch", 0)) + 1
        batch = state["val_batch"]
        if not total_batches:
            logger.info("Evaluation progress: batch=%s", batch)
            return
        interval = progress_interval(total_batches)
        if batch != 1 and batch != total_batches and batch % interval:
            return
        elapsed = time.perf_counter() - (state.get("val_start") or time.perf_counter())
        eta = (elapsed / batch) * max(total_batches - batch, 0) if batch else None
        logger.info("Evaluation progress: batch=%s/%s elapsed=%s eta=%s", batch, total_batches, format_seconds(elapsed), format_seconds(eta))

    def on_val_end(validator):
        elapsed = time.perf_counter() - (state.get("val_start") or time.perf_counter())
        metrics = getattr(validator, "metrics", None)
        logger.info("Evaluation finished: elapsed=%s metrics=%s", format_seconds(elapsed), to_serializable(metrics))

    def on_predict_start(predictor):
        state["predict_start"] = time.perf_counter()
        state["predict_batch"] = 0
        dataset = getattr(predictor, "dataset", None)
        state["predict_batches"] = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else None
        logger.info("Inference start: source=%s batches=%s", getattr(predictor.args, "source", args.source), state["predict_batches"])

    def on_predict_batch_end(predictor):
        total_batches = state.get("predict_batches")
        state["predict_batch"] = int(state.get("predict_batch", 0)) + 1
        batch = state["predict_batch"]
        if not total_batches:
            logger.info("Inference progress: batch=%s", batch)
            return
        interval = progress_interval(total_batches)
        if batch != 1 and batch != total_batches and batch % interval:
            return
        elapsed = time.perf_counter() - (state.get("predict_start") or time.perf_counter())
        eta = (elapsed / batch) * max(total_batches - batch, 0) if batch else None
        logger.info("Inference progress: batch=%s/%s elapsed=%s eta=%s", batch, total_batches, format_seconds(elapsed), format_seconds(eta))

    def on_predict_end(predictor):
        elapsed = time.perf_counter() - (state.get("predict_start") or time.perf_counter())
        logger.info("Inference finished: elapsed=%s save_dir=%s", format_seconds(elapsed), getattr(predictor, "save_dir", "unknown"))

    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_train_epoch_start": on_train_epoch_start,
        "on_train_batch_end": on_train_batch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
        "on_val_start": on_val_start,
        "on_val_batch_end": on_val_batch_end,
        "on_val_end": on_val_end,
        "on_predict_start": on_predict_start,
        "on_predict_batch_end": on_predict_batch_end,
        "on_predict_end": on_predict_end,
    }
    for event, callback in callbacks.items():
        if hasattr(model, "add_callback"):
            model.add_callback(event, callback)


def set_world_classes(model: Any, args: argparse.Namespace, logger: logging.Logger, include_unknown: bool = True) -> None:
    prompts = parse_csv(args.known_classes)
    if include_unknown:
        prompts.extend(parse_csv(args.unknown_prompts))
    if prompts and hasattr(model, "set_classes"):
        current_names = list(getattr(getattr(model, "model", None), "names", []) or [])
        if current_names == prompts:
            logger.info("YOLO-World prompts/classes already set to: %s", prompts)
            return
        try:
            model.set_classes(prompts)
        except RuntimeError as exc:
            if "Expected all tensors to be on the same device" not in str(exc):
                raise
            logger.warning("YOLO-World set_classes hit device mismatch. Retrying text embedding setup on CPU.")
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to("cpu")
            model.set_classes(prompts)
        logger.info("YOLO-World prompts/classes set to: %s", prompts)


def freeze_text_encoder(model: Any, logger: logging.Logger) -> int:
    frozen = 0
    module_keywords = ("text", "clip", "prompt")
    try:
        named_parameters = model.model.named_parameters()
    except Exception as exc:
        logger.warning("Could not inspect model parameters for text encoder freeze: %s", exc)
        return frozen
    for name, parameter in named_parameters:
        if any(keyword in name.lower() for keyword in module_keywords):
            parameter.requires_grad = False
            frozen += parameter.numel()
    logger.info("Frozen text/CLIP/prompt encoder parameters: %s", frozen)
    return frozen


def postprocess_prediction_results(predictions: Any, args: argparse.Namespace, output_dir: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    known = parse_csv(args.known_classes)
    unknown_start = len(known)
    prompts = known + parse_csv(args.unknown_prompts)
    rows: list[dict[str, Any]] = []
    for result in predictions or []:
        image_path = str(getattr(result, "path", ""))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        try:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
        except Exception:
            logger.warning("Could not read prediction boxes for %s", image_path)
            continue
        for box, conf, class_id_value in zip(xyxy, confs, classes):
            class_id = int(class_id_value)
            raw_label = prompts[class_id] if 0 <= class_id < len(prompts) else str(class_id)
            final_label = UNKNOWN_OBJECT_LABEL if class_id >= unknown_start else raw_label
            rows.append(
                {
                    "image": image_path,
                    "class_id": class_id,
                    "prompt_label": raw_label,
                    "final_label": final_label,
                    "confidence": float(conf),
                    "xyxy": [float(v) for v in box],
                    "is_unknown": class_id >= unknown_start,
                }
            )
    save_json(output_dir / "prediction_postprocessed.json", rows)
    logger.info("Post-processed predictions written to: %s", output_dir / "prediction_postprocessed.json")
    return rows


def postprocess_dual_prediction_results(
    known_predictions: Any,
    unknown_predictions: Any,
    args: argparse.Namespace,
    output_dir: Path,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    known = parse_csv(args.known_classes)
    unknown_prompts = parse_csv(args.unknown_prompts)
    rows: list[dict[str, Any]] = []
    for result in known_predictions or []:
        image_path = str(getattr(result, "path", ""))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box, conf, class_id_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            class_id = int(class_id_value)
            prompt_label = known[class_id] if 0 <= class_id < len(known) else str(class_id)
            rows.append(
                {
                    "image": image_path,
                    "class_id": class_id,
                    "prompt_label": prompt_label,
                    "final_label": prompt_label,
                    "confidence": float(conf),
                    "xyxy": [float(v) for v in box],
                    "is_unknown": False,
                    "source_model": "fine_tuned_known",
                }
            )
    for result in unknown_predictions or []:
        image_path = str(getattr(result, "path", ""))
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box, conf, class_id_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            class_id = int(class_id_value)
            prompt_label = unknown_prompts[class_id] if 0 <= class_id < len(unknown_prompts) else str(class_id)
            rows.append(
                {
                    "image": image_path,
                    "class_id": len(known) + class_id,
                    "prompt_label": prompt_label,
                    "final_label": UNKNOWN_OBJECT_LABEL,
                    "confidence": float(conf),
                    "xyxy": [float(v) for v in box],
                    "is_unknown": True,
                    "source_model": "pretrained_zero_shot_unknown",
                }
            )
    save_json(output_dir / "prediction_postprocessed.json", rows)
    logger.info(
        "Dual-model post-processed predictions written to: %s total=%s unknown=%s",
        output_dir / "prediction_postprocessed.json",
        len(rows),
        sum(1 for row in rows if row["is_unknown"]),
    )
    return rows


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    if args.timestamp_output and not args.experiment_name.startswith(datetime.now().strftime("%Y%m%d")):
        args.experiment_name = timestamped_name(args.experiment_name)
    experiment_dir = Path(args.output_dir) / args.experiment_name
    logger = setup_logger(experiment_dir)
    try:
        with redirect_console_to_file(experiment_dir) as console_log:
            logger.info("Command: %s", " ".join(sys.argv))
            logger.info("Parsed arguments: %s", vars(args))
            logger.info("Console stdout/stderr redirected to: %s", console_log)
            set_seed(args.seed)
            validate_device(str(args.device), logger)
            ensure_model_reference(args.model)
            versions = collect_versions()
            logger.info("Library versions: %s", versions)
            save_json(experiment_dir / "configs" / "args.json", {"args": vars(args), "versions": versions})

            from ultralytics import YOLOWorld

            logger.info("Stage: loading and preparing dataset yaml")
            data_yaml = prepare_known_dataset(args, experiment_dir, logger)
            logger.info("Stage finished: dataset ready")
            logger.info("Dataset yaml used: %s", data_yaml)
            logger.info("Model weight used: %s", args.model)
            logger.info("Output directory: %s", args.output_dir)
            logger.info("Experiment name: %s", args.experiment_name)

            logger.info("Stage: loading YOLO-World model")
            model = YOLOWorld(args.model)
            attach_progress_callbacks(model, args, logger)
            logger.info("Stage finished: YOLO-World model loaded")
            results: dict[str, Any] = {}
            if args.predict_only:
                if not args.source:
                    raise ValueError("--predict-only requires --source")
                logger.info("Stage: inference start")
                set_world_classes(model, args, logger, include_unknown=False)
                pred = model.predict(
                    source=args.source,
                    conf=args.conf_thres,
                    iou=args.iou_thres,
                    imgsz=args.imgsz,
                    device=args.device,
                    save=True,
                    project=str(experiment_dir),
                    name="predictions",
                    exist_ok=True,
                )
                results["prediction"] = to_serializable(pred)
                unknown_pred = []
                if args.use_zero_shot_unknown_model and parse_csv(args.unknown_prompts):
                    logger.info("Stage: zero-shot unknown inference start")
                    ensure_model_reference(args.zero_shot_unknown_model)
                    unknown_model = YOLOWorld(args.zero_shot_unknown_model)
                    unknown_args = argparse.Namespace(**vars(args))
                    unknown_args.known_classes = args.unknown_prompts
                    unknown_args.unknown_prompts = ""
                    set_world_classes(unknown_model, unknown_args, logger, include_unknown=False)
                    unknown_pred = unknown_model.predict(
                        source=args.source,
                        conf=args.unknown_conf_thres,
                        iou=args.iou_thres,
                        imgsz=args.imgsz,
                        device=args.device,
                        save=True,
                        project=str(experiment_dir),
                        name="predictions_unknown",
                        exist_ok=True,
                    )
                    results["prediction_unknown"] = to_serializable(unknown_pred)
                    results["prediction_postprocessed"] = postprocess_dual_prediction_results(pred, unknown_pred, args, experiment_dir / "predictions", logger)
                else:
                    results["prediction_postprocessed"] = postprocess_prediction_results(pred, args, experiment_dir / "predictions", logger)
                save_confidence_artifacts(results["prediction_postprocessed"], experiment_dir, logger)
                logger.info("Prediction output: %s", experiment_dir / "predictions")
                logger.info("Stage finished: inference complete")
            elif args.eval_only:
                logger.info("Stage: evaluation start split=%s", args.eval_split)
                set_world_classes(model, args, logger, include_unknown=True)
                metrics = model.val(data=str(data_yaml), split=args.eval_split, imgsz=args.imgsz, batch=args.batch_size, conf=args.conf_thres, iou=args.iou_thres, device=args.device)
                results["evaluation"] = to_serializable(metrics)
                save_json(experiment_dir / "metrics" / "evaluation.json", results["evaluation"])
                save_metrics_csv(experiment_dir / "metrics" / "final_metrics.csv", results["evaluation"], logger)
                logger.info("Evaluation metrics: %s", results["evaluation"])
                logger.info("Stage: sample visual evaluation start")
                logger.info("Reloading YOLO-World checkpoint for sample inference after validation.")
                sample_model = YOLOWorld(args.model)
                attach_progress_callbacks(sample_model, args, logger)
                results["sample_visualizations"] = save_sample_visualizations(sample_model, args, experiment_dir, logger)
                save_confidence_artifacts(results["sample_visualizations"], experiment_dir, logger)
                logger.info("Stage finished: evaluation complete")
            else:
                logger.info("Stage: training setup")
                train_args = dict(
                    data=str(data_yaml),
                    epochs=args.epochs,
                    batch=args.batch_size,
                    imgsz=args.imgsz,
                    lr0=args.lr0,
                    lrf=args.lrf,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    warmup_epochs=args.warmup_epochs,
                    optimizer=args.optimizer,
                    device=args.device,
                    workers=args.workers,
                    seed=args.seed,
                    amp=args.amp,
                    cache=args.cache,
                    resume=args.resume,
                    patience=args.patience,
                    save_period=args.save_period,
                    project=args.output_dir,
                    name=args.experiment_name,
                    exist_ok=True,
                )
                if args.freeze is not None:
                    train_args["freeze"] = args.freeze if len(args.freeze) != 1 else args.freeze[0]
                set_world_classes(model, args, logger, include_unknown=False)
                if args.freeze_text_encoder:
                    freeze_text_encoder(model, logger)
                logger.info("Training arguments: %s", train_args)
                logger.info("Stage: training start requested for epochs=%s", args.epochs)
                train_result = model.train(**train_args)
                results["training"] = to_serializable(train_result)
                logger.info("Training result: %s", results["training"])
                save_dir_value = getattr(train_result, "save_dir", None)
                save_dir = Path(save_dir_value) if save_dir_value else None
                copy_training_artifacts(save_dir, experiment_dir, logger)
                results["metrics_summary"] = log_epoch_metrics(experiment_dir, logger)
                if results["metrics_summary"]:
                    save_metrics_csv(experiment_dir / "metrics" / "final_metrics.csv", results["metrics_summary"].get("final_epoch"), logger)
                best = experiment_dir / "weights" / "best.pt"
                last = experiment_dir / "weights" / "last.pt"
                logger.info("Checkpoint best: %s exists=%s", best, best.exists())
                logger.info("Checkpoint last: %s exists=%s", last, last.exists())
                logger.info("Stage: sample visual evaluation start")
                results["sample_visualizations"] = save_sample_visualizations(model, args, experiment_dir, logger)
                save_confidence_artifacts(results["sample_visualizations"], experiment_dir, logger)
                logger.info("Stage finished: training complete")

            if args.export:
                logger.info("Stage: export start format=%s", args.export_format)
                export_result = model.export(format=args.export_format)
                results["export"] = to_serializable(export_result)
                logger.info("Export result: %s", results["export"])
                logger.info("Stage finished: export complete")
            save_json(experiment_dir / "configs" / "run_summary.json", results)
            elapsed_seconds = time.perf_counter() - start_time
            logger.info("Notebook finished. elapsed_seconds=%.3f experiment_dir=%s", elapsed_seconds, experiment_dir)
            return {"experiment_dir": str(experiment_dir), "results": results}
    except Exception:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error("Notebook failed. elapsed_seconds=%.3f experiment_dir=%s", elapsed_seconds, experiment_dir)
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)
