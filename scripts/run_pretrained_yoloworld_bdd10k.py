from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

from research_evaluation import (
    DEFAULT_KNOWN_CLASSES,
    DEFAULT_UNKNOWN_CLASSES,
    configure_file_logger,
    evaluate_research_metrics,
    infer_scale,
    load_yaml,
    normalize_names,
    parse_csv,
    save_json,
    save_summary_csv,
)


DEFAULT_PRETRAINED_MODELS = [
    "yolov8s-world.pt",
    "yolov8m-world.pt",
    "yolov8l-world.pt",
]

SCALE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "x": "X-Large", "n": "Nano"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pure pretrained YOLO-World inference/evaluation on BDD10K for Small, Medium, "
            "and Large models. This script does not train or fine-tune."
        )
    )
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    parser.add_argument("--output-dir", default="outputs/pretrained_yoloworld_bdd10k")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_PRETRAINED_MODELS),
        help="Comma-separated YOLO-World pretrained weights or local checkpoint paths.",
    )
    parser.add_argument("--known-classes", default=",".join(DEFAULT_KNOWN_CLASSES))
    parser.add_argument("--unknown-classes", default=",".join(DEFAULT_UNKNOWN_CLASSES))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.4f}"


def slugify(value: str) -> str:
    stem = Path(value).stem if not value.startswith(("http://", "https://")) else value.rsplit("/", 1)[-1].replace(".pt", "")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return slug or "pretrained_yoloworld"


def source_class_names(data_yaml: Path) -> list[str]:
    config = load_yaml(data_yaml)
    names = normalize_names(config.get("names", {}))
    return [name for _, name in sorted(names.items())]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_cells(row: dict[str, Any], prefix: str) -> dict[str, str]:
    return {
        "$mAP_{50}$": fmt(row.get(f"{prefix}_mAP50")),
        "$mAP_{50-95}$": fmt(row.get(f"{prefix}_mAP50_95")),
        "Precision": fmt(row.get(f"{prefix}_precision")),
        "Recall": fmt(row.get(f"{prefix}_recall")),
        "F1-Score": fmt(row.get(f"{prefix}_f1")),
    }


def sort_key(row: dict[str, Any]) -> int:
    return {"s": 0, "m": 1, "l": 2, "x": 3, "n": 4}.get(str(row.get("scale", "")), 99)


def build_pretrained_metrics_table(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=sort_key):
        all_item = {
            "Baseline": "YOLO-World Pretrained (No Fine-Tuning)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
            "Evaluation Target": "All Classes",
        }
        all_item.update(metric_cells(row, "all"))
        table_rows.append(all_item)

        unknown_item = {
            "Baseline": "YOLO-World Pretrained (No Fine-Tuning)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
            "Evaluation Target": "Unknown Classes",
        }
        unknown_item.update(metric_cells(row, "unknown"))
        table_rows.append(unknown_item)
    return table_rows


def per_class_map50(row: dict[str, Any], class_name: str) -> str:
    summary = row.get("research_summary") or {}
    unknown = (((summary.get("metrics") or {}).get("unknown_class") or {}).get("summary") or {})
    per_class = unknown.get("per_class") or {}
    return fmt((per_class.get(class_name) or {}).get("mAP50"))


def build_unknown_per_class_table(rows: list[dict[str, Any]], unknown_classes: list[str]) -> list[dict[str, str]]:
    by_scale = {row.get("scale"): row for row in rows}
    table_rows: list[dict[str, str]] = []
    for class_name in unknown_classes:
        table_rows.append(
            {
                "Unknown Classes": class_name,
                "Pretrained YW-Small": per_class_map50(by_scale.get("s", {}), class_name),
                "Pretrained YW-Medium": per_class_map50(by_scale.get("m", {}), class_name),
                "Pretrained YW-Large": per_class_map50(by_scale.get("l", {}), class_name),
            }
        )
    return table_rows


def build_efficiency_table(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    for row in sorted(rows, key=sort_key):
        summary = row.get("research_summary") or {}
        all_metrics = (summary.get("metrics") or {}).get("all_class") or {}
        metric_summary = all_metrics.get("summary") or {}
        speed = metric_summary.get("speed") or {}
        params = all_metrics.get("parameters")
        table_rows.append(
            {
                "Model": "YOLO-World Pretrained",
                "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
                "Parameters (M)": fmt((float(params) / 1_000_000.0) if params else None),
                "Inference Time (ms)": fmt(speed.get("inference") if isinstance(speed, dict) else None),
            }
        )
    return table_rows


def write_tables(out_dir: Path, rows: list[dict[str, Any]], unknown_classes: list[str]) -> None:
    metrics_headers = [
        "Baseline",
        "Scale",
        "Evaluation Target",
        "$mAP_{50}$",
        "$mAP_{50-95}$",
        "Precision",
        "Recall",
        "F1-Score",
    ]
    per_class_headers = ["Unknown Classes", "Pretrained YW-Small", "Pretrained YW-Medium", "Pretrained YW-Large"]
    efficiency_headers = ["Model", "Scale", "Parameters (M)", "Inference Time (ms)"]
    table_specs = [
        ("table6_pretrained_yoloworld", build_pretrained_metrics_table(rows), metrics_headers),
        ("table7_pretrained_yoloworld_unknown_per_class_map50", build_unknown_per_class_table(rows, unknown_classes), per_class_headers),
        ("table8_pretrained_yoloworld_efficiency_latency", build_efficiency_table(rows), efficiency_headers),
    ]
    for name, table_rows, headers in table_specs:
        write_csv(out_dir / f"{name}.csv", table_rows, headers)
        write_md_table(out_dir / f"{name}.md", table_rows, headers)


def write_report(out_dir: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Pretrained YOLO-World BDD10K Evaluation",
        "",
        "This output evaluates pure pretrained YOLO-World models without training or fine-tuning.",
        "",
        f"Evaluated models: {len(rows)}",
        f"Failures: {len(failures)}",
        "",
        "## Outputs",
        "",
        "- `pretrained_yoloworld_summary.csv` / `pretrained_yoloworld_summary.json`",
        "- `table6_pretrained_yoloworld.csv` / `table6_pretrained_yoloworld.md`",
        "- `table7_pretrained_yoloworld_unknown_per_class_map50.csv` / `table7_pretrained_yoloworld_unknown_per_class_map50.md`",
        "- `table8_pretrained_yoloworld_efficiency_latency.csv` / `table8_pretrained_yoloworld_efficiency_latency.md`",
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('model')}`: {failure.get('error')}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_file_logger(out_dir / "pretrained_yoloworld_eval.log")
    data_yaml = Path(args.data_yaml)
    models = parse_csv(args.models)
    known_classes = parse_csv(args.known_classes)
    unknown_classes = parse_csv(args.unknown_classes)
    all_classes = source_class_names(data_yaml)

    save_json(
        out_dir / "config.json",
        {
            "data_yaml": str(data_yaml),
            "models": models,
            "known_classes": known_classes,
            "unknown_classes": unknown_classes,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch_size": args.batch_size,
            "conf_thres": args.conf_thres,
            "iou_thres": args.iou_thres,
            "device": args.device,
            "workers": args.workers,
            "note": "Pure pretrained YOLO-World evaluation. No training or fine-tuning is performed.",
        },
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_name in models:
        model_dir = out_dir / slugify(model_name)
        scale = infer_scale(model_dir, Path(model_name), {"model": model_name})
        try:
            from ultralytics import YOLOWorld

            logger.info("Loading pretrained YOLO-World model=%s scale=%s", model_name, scale)
            model = YOLOWorld(model_name)
            result = evaluate_research_metrics(
                model=model,
                model_type="yoloworld",
                source_data_yaml=data_yaml,
                output_dir=model_dir / "research_eval",
                trained_classes=all_classes,
                known_classes=known_classes,
                unknown_classes=unknown_classes,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch_size,
                conf=args.conf_thres,
                iou=args.iou_thres,
                device=args.device,
                workers=args.workers,
                run_dir=model_dir,
                checkpoint=Path(model_name),
                scale=scale,
                training_config="pretrained",
                logger=logger,
            )
            row = dict(result.get("summary_row") or {})
            row["research_summary"] = result
            rows.append(row)
        except Exception as exc:
            logger.exception("Pretrained YOLO-World evaluation failed for model=%s", model_name)
            failures.append({"model": model_name, "error": str(exc)})

    rows.sort(key=sort_key)
    save_summary_csv(out_dir / "pretrained_yoloworld_summary.csv", rows)
    save_json(out_dir / "pretrained_yoloworld_summary.json", {"rows": rows, "failures": failures})
    write_tables(out_dir, rows, unknown_classes)
    write_report(out_dir, rows, failures)

    print(f"Evaluated pretrained YOLO-World models: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {out_dir}")
    if failures:
        raise RuntimeError(f"{len(failures)} pretrained model(s) failed. See {out_dir / 'pretrained_yoloworld_summary.json'}")


if __name__ == "__main__":
    main()
