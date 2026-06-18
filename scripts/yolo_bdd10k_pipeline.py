from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def timestamped_name(name: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate/predict standard YOLO on BDD10K with Ultralytics.")
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--output-dir", default="runs/yolo_bdd10k")
    parser.add_argument("--experiment-name", default="yolo_bdd10k_finetune")
    parser.add_argument("--timestamp-output", action="store_true")
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
    parser.add_argument("--eval-split", choices=("train", "val"), default="val")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--source", default="")
    parser.add_argument("--save-eval-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample-source", default="")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--export-format", default="onnx")
    return parser


def setup_logger(experiment_dir: Path) -> logging.Logger:
    (experiment_dir / "logs").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s/%(funcName)s | %(message)s")
    logger = logging.getLogger("yolo_bdd10k")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    for handler in (
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
        logging.FileHandler(experiment_dir / "logs" / "train.log", mode="a", encoding="utf-8"),
    ):
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


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise ValueError("data yaml must define names as a dict or list")


def resolve_split_path(data: dict[str, Any], split: str) -> Path:
    root = Path(data.get("path", "."))
    value = data.get(split)
    if not value:
        raise ValueError(f"Dataset yaml does not define split: {split}")
    path = Path(value)
    return path if path.is_absolute() else root / path


def find_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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


def ensure_model_reference(model: str) -> None:
    if Path(model).exists():
        return
    if model.startswith(("yolo", "http://", "https://")):
        return
    raise FileNotFoundError(
        f"Model '{model}' does not exist locally and does not look like an Ultralytics model name. "
        "Use e.g. yolov8s.pt, yolov8m.pt, yolov8l.pt, or a valid checkpoint path."
    )


def copy_training_artifacts(save_dir: Path | None, experiment_dir: Path, logger: logging.Logger) -> None:
    metrics_dir = experiment_dir / "metrics"
    configs_dir = experiment_dir / "configs"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    source = save_dir if save_dir and save_dir.exists() else experiment_dir
    for src_name, dst in {
        "results.csv": metrics_dir / "training_history.csv",
        "args.yaml": configs_dir / "ultralytics_args.yaml",
    }.items():
        src = source / src_name
        if src.exists():
            shutil.copy2(src, dst)
            logger.info("Copied artifact: %s -> %s", src, dst)
            if source.resolve() == experiment_dir.resolve():
                src.unlink()
    weights = source / "weights"
    if weights.exists():
        shutil.copytree(weights, experiment_dir / "weights", dirs_exist_ok=True)
        logger.info("Copied weights: %s -> %s", weights, experiment_dir / "weights")


def log_epoch_metrics(experiment_dir: Path, logger: logging.Logger) -> dict[str, Any] | None:
    results_csv = experiment_dir / "metrics" / "training_history.csv"
    if not results_csv.exists():
        logger.warning("Training history not found: %s", results_csv)
        return None
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        logger.warning("Training history is empty: %s", results_csv)
        return None
    for row in rows:
        epoch = row.get("epoch", "?")
        keys = [
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
        ]
        logger.info("Epoch %s metrics: %s", epoch, {key: row[key] for key in keys if key in row})
    summary = {"results_csv": str(results_csv), "num_epochs_logged": len(rows), "final_epoch": rows[-1]}
    save_json(experiment_dir / "metrics" / "metrics_summary.json", summary)
    return summary


def save_metrics_csv(path: Path, metrics: Any, logger: logging.Logger) -> None:
    data = to_serializable(metrics)
    rows = [{"metric": key, "value": value} for key, value in data.items()] if isinstance(data, dict) else [{"metric": "result", "value": data}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved metrics csv: %s", path)


def draw_labeled_box(image: Any, box: list[float], label: str, color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(label, font, 0.55, 1)
    y_text = max(y1, th + base + 4)
    cv2.rectangle(image, (x1, y_text - th - base - 4), (x1 + tw + 6, y_text + base), color, -1)
    cv2.putText(image, label, (x1 + 3, y_text - 3), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = values
    return [(xc - bw / 2) * width, (yc - bh / 2) * height, (xc + bw / 2) * width, (yc + bh / 2) * height]


def ground_truth_label_path(image_path: Path) -> Path:
    text = str(image_path)
    if "/images/" in text:
        return Path(text.replace("/images/", "/labels/")).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def draw_ground_truth(image: Any, image_path: Path, names: dict[int, str], logger: logging.Logger) -> int:
    label_path = ground_truth_label_path(image_path)
    if not label_path.exists():
        logger.warning("Ground-truth label not found for sample image: %s", label_path)
        return 0
    height, width = image.shape[:2]
    count = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        box = yolo_to_xyxy([float(v) for v in parts[1:]], width, height)
        draw_labeled_box(image, box, names.get(class_id, str(class_id)), (255, 120, 0))
        count += 1
    return count


def add_panel_title(image: Any, title: str) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def save_confidence_histogram(rows: list[dict[str, Any]], experiment_dir: Path, logger: logging.Logger) -> None:
    values = []
    for row in rows:
        try:
            values.append(float(row["confidence"]))
        except Exception:
            pass
    metrics_dir = experiment_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    bins = [i / 10 for i in range(11)]
    csv_rows = []
    for start, end in zip(bins[:-1], bins[1:]):
        count = sum(1 for v in values if start <= v <= end) if end == 1.0 else sum(1 for v in values if start <= v < end)
        csv_rows.append({"confidence_min": f"{start:.1f}", "confidence_max": f"{end:.1f}", "count": count})
    with (metrics_dir / "confidence_histogram.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["confidence_min", "confidence_max", "count"])
        writer.writeheader()
        writer.writerows(csv_rows)
    if not values:
        logger.warning("No detections available for confidence histogram.")
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(values, bins=bins, color="#2f6fdd", edgecolor="black", alpha=0.8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Detection count")
        ax.set_title("Confidence Histogram on Evaluation Samples")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(metrics_dir / "confidence_histogram.png", dpi=160)
        plt.close(fig)
        logger.info("Saved confidence histogram: %s", metrics_dir / "confidence_histogram.png")
    except Exception as exc:
        logger.warning("Could not save confidence histogram image: %s", exc)


def save_sample_visualizations(model: Any, args: argparse.Namespace, experiment_dir: Path, data: dict[str, Any], logger: logging.Logger) -> list[dict[str, Any]]:
    if not args.save_eval_samples:
        return []
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python is not available; skipping sample visualizations.")
        return []
    names = normalize_names(data["names"])
    source = Path(args.sample_source) if args.sample_source else resolve_split_path(data, "val")
    images = find_images(source)[: args.sample_count]
    if not images:
        logger.warning("No sample images found: %s", source)
        return []
    output_dir = experiment_dir / "evaluation" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Sample evaluation inference start: source=%s count=%s", source, len(images))
    predictions = model.predict(
        source=[str(p) for p in images],
        conf=args.conf_thres,
        iou=args.iou_thres,
        imgsz=args.imgsz,
        device=args.device,
        save=False,
        verbose=False,
    )
    rows: list[dict[str, Any]] = []
    for idx, result in enumerate(predictions):
        image_path = images[idx]
        original = cv2.imread(str(image_path))
        if original is None:
            logger.warning("Could not read sample image: %s", image_path)
            continue
        gt_panel = original.copy()
        pred_panel = original.copy()
        gt_count = draw_ground_truth(gt_panel, image_path, names, logger)
        pred_count = 0
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                label = f"{names.get(class_id, str(class_id))} {conf:.2f}"
                draw_labeled_box(pred_panel, xyxy, label, (0, 180, 80))
                rows.append(
                    {
                        "image": str(image_path),
                        "class_id": class_id,
                        "label": names.get(class_id, str(class_id)),
                        "confidence": conf,
                        "xyxy": xyxy,
                    }
                )
                pred_count += 1
        add_panel_title(gt_panel, f"Ground Truth ({gt_count})")
        add_panel_title(pred_panel, f"Inference ({pred_count})")
        combined = cv2.hconcat([gt_panel, pred_panel])
        out_path = output_dir / f"{image_path.stem}_gt_vs_pred.jpg"
        cv2.imwrite(str(out_path), combined)
    save_json(experiment_dir / "evaluation" / "sample_predictions.json", rows)
    save_confidence_histogram(rows, experiment_dir, logger)
    logger.info("Saved sample visualizations: %s", output_dir)
    return rows


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    experiment_name = timestamped_name(args.experiment_name) if args.timestamp_output else args.experiment_name
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(experiment_dir)
    results: dict[str, Any] = {"experiment_dir": str(experiment_dir)}
    try:
        with redirect_console_to_file(experiment_dir) as console_log:
            logger.info("Command: %s", " ".join(sys.argv))
            logger.info("Console log: %s", console_log)
            logger.info("YOLO BDD10K pipeline started.")
            validate_device(args.device, logger)
            ensure_model_reference(args.model)
            set_seed(args.seed)
            data = load_yaml(Path(args.data_yaml))
            names = normalize_names(data.get("names"))
            if "train" not in data or "val" not in data:
                raise ValueError("Dataset yaml must contain train and val splits.")
            logger.info("Dataset yaml: %s", args.data_yaml)
            logger.info("Dataset train: %s", resolve_split_path(data, "train"))
            logger.info("Dataset val: %s", resolve_split_path(data, "val"))
            logger.info("Class names: %s", names)
            logger.info("Model: %s", args.model)
            logger.info("Output directory: %s", experiment_dir)
            logger.info("Training config: epochs=%s batch=%s imgsz=%s lr0=%s optimizer=%s amp=%s resume=%s", args.epochs, args.batch_size, args.imgsz, args.lr0, args.optimizer, args.amp, args.resume)
            save_json(experiment_dir / "configs" / "args.json", vars(args))
            save_json(experiment_dir / "configs" / "versions.json", collect_versions())
            (experiment_dir / "configs" / "config_used.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            from ultralytics import YOLO

            model = YOLO(args.model)
            if args.predict_only:
                if not args.source:
                    raise ValueError("--predict-only requires --source")
                logger.info("Prediction start: source=%s", args.source)
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
            elif args.eval_only:
                logger.info("Evaluation start: split=%s", args.eval_split)
                metrics = model.val(
                    data=args.data_yaml,
                    split=args.eval_split,
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    conf=args.conf_thres,
                    iou=args.iou_thres,
                    device=args.device,
                    workers=args.workers,
                    project=str(experiment_dir),
                    name="validation",
                    exist_ok=True,
                )
                results["evaluation"] = to_serializable(metrics)
                save_json(experiment_dir / "metrics" / "evaluation.json", results["evaluation"])
                save_metrics_csv(experiment_dir / "metrics" / "final_metrics.csv", metrics, logger)
                save_sample_visualizations(model, args, experiment_dir, data, logger)
            else:
                logger.info("Training start: epochs=%s", args.epochs)
                train_args = {
                    "data": args.data_yaml,
                    "epochs": args.epochs,
                    "batch": args.batch_size,
                    "imgsz": args.imgsz,
                    "lr0": args.lr0,
                    "lrf": args.lrf,
                    "weight_decay": args.weight_decay,
                    "momentum": args.momentum,
                    "warmup_epochs": args.warmup_epochs,
                    "optimizer": args.optimizer,
                    "device": args.device,
                    "workers": args.workers,
                    "seed": args.seed,
                    "amp": args.amp,
                    "cache": args.cache,
                    "resume": args.resume,
                    "freeze": args.freeze,
                    "patience": args.patience,
                    "save_period": args.save_period,
                    "project": args.output_dir,
                    "name": experiment_name,
                    "exist_ok": True,
                }
                train_result = model.train(**train_args)
                results["training"] = to_serializable(train_result)
                copy_training_artifacts(Path(getattr(train_result, "save_dir", experiment_dir)), experiment_dir, logger)
                results["training_history"] = log_epoch_metrics(experiment_dir, logger)
                logger.info("Evaluation start after training: split=val")
                metrics = model.val(
                    data=args.data_yaml,
                    split="val",
                    imgsz=args.imgsz,
                    batch=args.batch_size,
                    conf=args.conf_thres,
                    iou=args.iou_thres,
                    device=args.device,
                    workers=args.workers,
                    project=str(experiment_dir),
                    name="validation",
                    exist_ok=True,
                )
                results["evaluation"] = to_serializable(metrics)
                save_json(experiment_dir / "metrics" / "evaluation.json", results["evaluation"])
                save_metrics_csv(experiment_dir / "metrics" / "final_metrics.csv", metrics, logger)
                save_sample_visualizations(model, args, experiment_dir, data, logger)

            if args.export:
                logger.info("Export start: format=%s", args.export_format)
                results["export"] = to_serializable(model.export(format=args.export_format))

            elapsed = time.time() - start
            results["elapsed_seconds"] = elapsed
            save_json(experiment_dir / "configs" / "run_summary.json", results)
            logger.info("Pipeline finished in %.2f seconds.", elapsed)
            logger.info("Notebook finished marker: elapsed_seconds=%.2f", elapsed)
            return results
    except Exception:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
