from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from yoloworld_bdd10k_pipeline import (
    BDD10K_NAMES,
    add_common_args,
    attach_progress_callbacks,
    auto_convert_bdd_json_split,
    collect_versions,
    copy_training_artifacts,
    count_yolo_annotations,
    ensure_model_reference,
    find_images,
    freeze_text_encoder,
    image_size,
    label_path_for_image,
    load_yaml,
    log_epoch_metrics,
    normalize_names,
    parse_csv,
    redirect_console_to_file,
    resolve_split_path,
    run_research_evaluation,
    save_json,
    set_seed,
    set_world_classes,
    timestamped_name,
    to_serializable,
    validate_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train YOLO-World on known BDD10K classes while retaining open-vocabulary behavior "
            "using pseudo-labels from a frozen pretrained YOLO-World teacher."
        )
    )
    add_common_args(parser)
    parser.set_defaults(
        experiment_name="yoloworld_bdd10k_retention_finetune",
        unknown_conf_thres=0.25,
        use_zero_shot_unknown_model=False,
        research_eval=True,
    )
    parser.add_argument(
        "--teacher-model",
        default="",
        help="Frozen YOLO-World teacher used to generate unknown pseudo-labels. Default: same as --model.",
    )
    parser.add_argument(
        "--pseudo-splits",
        default="train",
        help="Comma-separated splits that receive teacher unknown pseudo-labels. Default: train.",
    )
    parser.add_argument(
        "--pseudo-batch-size",
        type=int,
        default=16,
        help="Teacher inference batch size for pseudo-label generation.",
    )
    parser.add_argument(
        "--pseudo-conf-thres",
        type=float,
        default=0.25,
        help="Teacher confidence threshold for unknown pseudo-labels.",
    )
    parser.add_argument(
        "--pseudo-iou-thres",
        type=float,
        default=0.7,
        help="Teacher NMS IoU threshold for unknown pseudo-labels.",
    )
    parser.add_argument(
        "--pseudo-known-iou-suppress",
        type=float,
        default=0.30,
        help="Drop unknown pseudo-labels that overlap known GT boxes above this IoU.",
    )
    parser.add_argument(
        "--pseudo-max-per-image",
        type=int,
        default=60,
        help="Maximum unknown pseudo-labels kept per image after filtering.",
    )
    parser.add_argument(
        "--no-pseudo-labels",
        action="store_true",
        help="Build a 10-prompt retention dataset without teacher pseudo-labels; useful for ablations.",
    )
    return parser


def setup_logger(experiment_dir: Path) -> logging.Logger:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    root_log = Path("retention_training.log")
    exp_log = experiment_dir / "logs" / "retention_train.log"
    exp_log.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s/%(funcName)s | %(message)s")
    logger = logging.getLogger("yoloworld_bdd10k_retention")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    for handler in (logging.FileHandler(root_log, mode="w"), logging.FileHandler(exp_log, mode="a")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def yolo_to_xyxy(values: list[float], width: int, height: int) -> list[float]:
    xc, yc, bw, bh = values
    return [
        (xc - bw / 2.0) * width,
        (yc - bh / 2.0) * height,
        (xc + bw / 2.0) * width,
        (yc + bh / 2.0) * height,
    ]


def xyxy_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height, (x2 - x1) / width, (y2 - y1) / height)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def read_known_gt(
    label_path: Path,
    old_to_new: dict[int, int],
    width: int,
    height: int,
) -> tuple[list[str], list[list[float]]]:
    lines: list[str] = []
    boxes: list[list[float]] = []
    if not label_path.exists():
        return lines, boxes
    for row in label_path.read_text(encoding="utf-8").splitlines():
        parts = row.split()
        if len(parts) < 5:
            continue
        old_id = int(float(parts[0]))
        if old_id not in old_to_new:
            continue
        values = [float(value) for value in parts[1:5]]
        lines.append(" ".join([str(old_to_new[old_id]), *[f"{value:.6f}" for value in values]]))
        boxes.append(yolo_to_xyxy(values, width, height))
    return lines, boxes


def set_teacher_unknown_classes(model: Any, unknown_prompts: list[str], device: str, logger: logging.Logger) -> None:
    if not unknown_prompts:
        return
    try:
        model.set_classes(unknown_prompts)
    except RuntimeError as exc:
        if "Expected all tensors to be on the same device" not in str(exc):
            raise
        if hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to("cpu")
        model.set_classes(unknown_prompts)
    if device and device not in ("cpu", "mps") and hasattr(model, "model") and hasattr(model.model, "to"):
        target = f"cuda:{device}" if str(device).isdigit() else device
        model.model.to(target)
    logger.info("Teacher unknown prompts set: %s", unknown_prompts)


def predict_unknown_pseudo_labels(
    teacher: Any,
    images: list[Path],
    unknown_prompts: list[str],
    unknown_offset: int,
    known_boxes_by_image: dict[Path, list[list[float]]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[Path, list[str]]:
    if args.no_pseudo_labels or not images or not unknown_prompts:
        return {image: [] for image in images}
    output: dict[Path, list[str]] = {}
    total_kept = 0
    total_dropped_overlap = 0
    for start in range(0, len(images), args.pseudo_batch_size):
        batch = images[start : start + args.pseudo_batch_size]
        logger.info("Teacher pseudo-label inference: %s/%s", min(start + len(batch), len(images)), len(images))
        results = teacher.predict(
            source=[str(path) for path in batch],
            conf=args.pseudo_conf_thres,
            iou=args.pseudo_iou_thres,
            imgsz=args.imgsz,
            device=args.device,
            save=False,
            verbose=False,
        )
        for image, result in zip(batch, results):
            width, height = image_size(image)
            known_boxes = known_boxes_by_image.get(image, [])
            rows: list[tuple[float, str]] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for xyxy, conf, class_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
                    class_id = int(class_value)
                    if class_id < 0 or class_id >= len(unknown_prompts):
                        continue
                    if any(box_iou([float(v) for v in xyxy], gt_box) >= args.pseudo_known_iou_suppress for gt_box in known_boxes):
                        total_dropped_overlap += 1
                        continue
                    yolo_box = xyxy_to_yolo([float(v) for v in xyxy], width, height)
                    if yolo_box is None:
                        continue
                    new_class_id = unknown_offset + class_id
                    line = f"{new_class_id} " + " ".join(f"{value:.6f}" for value in yolo_box)
                    rows.append((float(conf), line))
            rows.sort(key=lambda item: item[0], reverse=True)
            kept = [line for _, line in rows[: args.pseudo_max_per_image]]
            output[image] = kept
            total_kept += len(kept)
    logger.info(
        "Teacher pseudo-label generation finished: images=%s kept=%s dropped_known_overlap=%s",
        len(images),
        total_kept,
        total_dropped_overlap,
    )
    return output


def prepare_retention_dataset(args: argparse.Namespace, experiment_dir: Path, logger: logging.Logger) -> Path:
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
    unknown = parse_csv(args.unknown_classes) or parse_csv(args.unknown_prompts) or [name for name in names.values() if name not in known]
    if not known:
        raise ValueError("--known-classes must contain at least one class")
    missing_known = [name for name in known if name not in name_to_id]
    if missing_known:
        raise ValueError(f"Known classes not present in data yaml: {missing_known}")
    old_to_new = {name_to_id[name]: idx for idx, name in enumerate(known)}
    train_names = known + unknown
    out_root = experiment_dir / "dataset_retention"
    yaml_out = experiment_dir / "configs" / "config_retention.yaml"
    out_root.mkdir(parents=True, exist_ok=True)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    pseudo_splits = set(parse_csv(args.pseudo_splits))

    teacher = None
    if not args.no_pseudo_labels and pseudo_splits:
        from ultralytics import YOLOWorld

        teacher_weight = args.teacher_model or args.model
        ensure_model_reference(teacher_weight)
        logger.info("Loading frozen teacher for retention pseudo-labels: %s", teacher_weight)
        teacher = YOLOWorld(teacher_weight)
        set_teacher_unknown_classes(teacher, unknown, args.device, logger)

    dataset_summary: dict[str, Any] = {"known_classes": known, "unknown_prompts": unknown, "splits": {}}
    for split in ("train", "val"):
        image_dir = resolve_split_path(dataset_root, config.get(split))
        if image_dir is None:
            continue
        label_dir = Path(str(image_dir).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"))
        auto_convert_bdd_json_split(dataset_root, image_dir, label_dir, split, names, logger)
        images = find_images(image_dir)
        out_img_dir = out_root / "images" / split
        out_label_dir = out_root / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        known_lines_by_image: dict[Path, list[str]] = {}
        known_boxes_by_image: dict[Path, list[list[float]]] = {}
        for image in images:
            target_image = out_img_dir / image.relative_to(image_dir)
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if not target_image.exists():
                try:
                    target_image.symlink_to(image.resolve())
                except OSError:
                    shutil.copy2(image, target_image)
            width, height = image_size(image)
            source_label = label_path_for_image(image, image_dir, label_dir)
            known_lines, known_boxes = read_known_gt(source_label, old_to_new, width, height)
            known_lines_by_image[image] = known_lines
            known_boxes_by_image[image] = known_boxes

        pseudo_by_image = {image: [] for image in images}
        if teacher is not None and split in pseudo_splits:
            pseudo_by_image = predict_unknown_pseudo_labels(
                teacher=teacher,
                images=images,
                unknown_prompts=unknown,
                unknown_offset=len(known),
                known_boxes_by_image=known_boxes_by_image,
                args=args,
                logger=logger,
            )
        for image in images:
            target_label = out_label_dir / image.relative_to(image_dir).with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            rows = known_lines_by_image.get(image, []) + pseudo_by_image.get(image, [])
            target_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        known_count = sum(len(rows) for rows in known_lines_by_image.values())
        pseudo_count = sum(len(rows) for rows in pseudo_by_image.values())
        dataset_summary["splits"][split] = {"images": len(images), "known_gt_labels": known_count, "unknown_pseudo_labels": pseudo_count}
        logger.info("Retention split ready: split=%s images=%s known_gt=%s unknown_pseudo=%s", split, len(images), known_count, pseudo_count)

    train_annotation_count = count_yolo_annotations(out_root / "labels" / "train")
    val_annotation_count = count_yolo_annotations(out_root / "labels" / "val")
    if train_annotation_count == 0:
        raise RuntimeError("Retention dataset has no train annotations. Check data paths and --known-classes.")
    if val_annotation_count == 0:
        raise RuntimeError("Retention dataset has no val annotations. Check data paths and --known-classes.")

    retention_config = {"path": str(out_root.resolve()), "train": "images/train", "val": "images/val", "names": {idx: name for idx, name in enumerate(train_names)}}
    yaml_out.write_text(yaml.safe_dump(retention_config, sort_keys=False), encoding="utf-8")
    save_json(experiment_dir / "configs" / "retention_dataset_summary.json", dataset_summary)
    logger.info("Retention dataset yaml: %s", yaml_out)
    logger.info("Retention dataset summary: %s", dataset_summary)
    return yaml_out


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
            if args.teacher_model:
                ensure_model_reference(args.teacher_model)
            versions = collect_versions()
            logger.info("Library versions: %s", versions)
            save_json(experiment_dir / "configs" / "args.json", {"args": vars(args), "versions": versions})

            from ultralytics import YOLOWorld

            logger.info("Stage: building retention dataset")
            data_yaml = prepare_retention_dataset(args, experiment_dir, logger)
            logger.info("Stage finished: retention dataset ready")

            logger.info("Stage: loading YOLO-World student")
            model = YOLOWorld(args.model)
            attach_progress_callbacks(model, args, logger)
            set_world_classes(model, args, logger, include_unknown=True)
            if args.freeze_text_encoder:
                freeze_text_encoder(model, logger)

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
            logger.info("Training arguments: %s", train_args)
            train_result = model.train(**train_args)
            results: dict[str, Any] = {"training": to_serializable(train_result)}
            save_dir_value = getattr(train_result, "save_dir", None)
            copy_training_artifacts(Path(save_dir_value) if save_dir_value else None, experiment_dir, logger)
            results["metrics_summary"] = log_epoch_metrics(experiment_dir, logger)
            if args.research_eval:
                checkpoint = experiment_dir / "weights" / "best.pt"
                if checkpoint.exists():
                    logger.info("Reloading best checkpoint for research evaluation: %s", checkpoint)
                    model = YOLOWorld(str(checkpoint))
                    set_world_classes(model, args, logger, include_unknown=True)
                results["research_evaluation"] = run_research_evaluation(model, args, experiment_dir, data_yaml, logger)
            save_json(experiment_dir / "configs" / "run_summary.json", results)
            elapsed_seconds = time.perf_counter() - start_time
            logger.info("Retention pipeline finished. elapsed_seconds=%.3f experiment_dir=%s", elapsed_seconds, experiment_dir)
            return {"experiment_dir": str(experiment_dir), "results": results}
    except Exception:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error("Retention pipeline failed. elapsed_seconds=%.3f experiment_dir=%s", elapsed_seconds, experiment_dir)
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
