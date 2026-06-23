from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from research_evaluation import (
    DEFAULT_UNKNOWN_CLASSES,
    configure_file_logger,
    discover_run_dirs,
    evaluate_run_dir,
    infer_model_type,
    parse_csv,
    read_run_args,
    save_json,
    save_summary_csv,
)


SCALE_LABEL = {"s": "Small", "m": "Medium", "l": "Large", "x": "X-Large", "n": "Nano"}
MODEL_LABEL = {"yoloworld": "YOLO-World", "yolo": "Standard YOLO"}
SCHEME = {
    ("yoloworld", "all_class"): "Scheme 1",
    ("yoloworld", "known_class"): "Scheme 2",
    ("yolo", "all_class"): "Scheme 3",
    ("yolo", "known_class"): "Scheme 4",
    ("yoloworld", "pretrained"): "Pretrained Baseline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and compile all evaluations required by results.md.")
    parser.add_argument("--runs-root", action="append", default=["runs/yolo_bdd10k", "runs/yoloworld_bdd10k"])
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    parser.add_argument("--output-dir", default="outputs/results_eval")
    parser.add_argument("--known-classes", default="car,bus,truck")
    parser.add_argument("--unknown-classes", default=",".join(DEFAULT_UNKNOWN_CLASSES))
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--research-output-name", default="research_eval")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true", help="Only compile existing research_eval artifacts.")
    parser.add_argument("--allow-failures", action="store_true", help="Write failure report instead of exiting non-zero.")
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_md_table(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_or_eval_run(run_dir: Path, args: argparse.Namespace, logger: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not args.skip_eval:
        try:
            result = evaluate_run_dir(
                run_dir=run_dir,
                source_data_yaml=Path(args.data_yaml),
                output_name=args.research_output_name,
                model_type="auto",
                known_classes=parse_csv(args.known_classes),
                unknown_classes=parse_csv(args.unknown_classes),
                checkpoint_name=args.checkpoint_name,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch_size,
                conf=args.conf_thres,
                iou=args.iou_thres,
                device=args.device,
                workers=args.workers,
                logger=logger,
            )
            return result, None
        except Exception as exc:
            logger.exception("Failed evaluating %s", run_dir)
            return None, {"run_dir": str(run_dir), "error": str(exc)}

    summary = read_json(run_dir / args.research_output_name / "research_metrics_summary.json")
    if not summary:
        return None, {"run_dir": str(run_dir), "error": "missing research_metrics_summary.json"}
    return summary, None


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    row = dict(result.get("summary_row") or {})
    row["research_summary"] = result
    return row


def scale_sort(row: dict[str, Any]) -> int:
    return {"s": 0, "m": 1, "l": 2, "x": 3, "n": 4}.get(str(row.get("scale", "")), 99)


def model_sort(row: dict[str, Any]) -> tuple[int, int, int]:
    model_order = {"yoloworld": 0, "yolo": 1}.get(str(row.get("model_type", "")), 9)
    train_order = {"all_class": 0, "known_class": 1, "pretrained": 2}.get(str(row.get("training_config", "")), 9)
    return model_order, train_order, scale_sort(row)


def metric_row(row: dict[str, Any], prefix: str) -> dict[str, str]:
    return {
        "$mAP_{50}$": fmt(row.get(f"{prefix}_mAP50")),
        "$mAP_{50-95}$": fmt(row.get(f"{prefix}_mAP50_95")),
        "mAP": fmt(row.get(f"{prefix}_mAP")),
        "mIoU": fmt(row.get(f"{prefix}_mIoU")),
        "Precision": fmt(row.get(f"{prefix}_precision")),
        "Recall": fmt(row.get(f"{prefix}_recall")),
        "F1-Score": fmt(row.get(f"{prefix}_f1")),
    }


def build_table1(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(rows, key=model_sort):
        if row.get("training_config") != "all_class":
            continue
        item = {
            "Scheme ID": SCHEME.get((row.get("model_type"), row.get("training_config")), ""),
            "Model & Training Config": f"{MODEL_LABEL.get(row.get('model_type'), row.get('model_type'))} (All Class)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
        }
        item.update(metric_row(row, "all"))
        output.append(item)
    return output


def build_table2(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(rows, key=model_sort):
        if row.get("training_config") != "known_class":
            continue
        item = {
            "Scheme ID": SCHEME.get((row.get("model_type"), row.get("training_config")), ""),
            "Model & Training Config": f"{MODEL_LABEL.get(row.get('model_type'), row.get('model_type'))} (Known Class)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
            "Evaluation Target": "Unknown Classes (Zero-Shot)",
        }
        if row.get("model_type") == "yolo":
            item.update({"$mAP_{50}$": "0.0000", "$mAP_{50-95}$": "0.0000", "mAP": "0.0000", "mIoU": "0.0000", "Precision": "0.0000", "Recall": "0.0000", "F1-Score": "0.0000"})
        else:
            item.update(metric_row(row, "unknown"))
        output.append(item)
    return output


def per_class_map50(row: dict[str, Any], class_name: str) -> str:
    summary = row.get("research_summary") or {}
    unknown = (((summary.get("metrics") or {}).get("unknown_class") or {}).get("summary") or {})
    per_class = unknown.get("per_class") or {}
    return fmt((per_class.get(class_name) or {}).get("mAP50"))


def build_table3(rows: list[dict[str, Any]], unknown_classes: list[str]) -> list[dict[str, str]]:
    by_key = {(row.get("model_type"), row.get("training_config"), row.get("scale")): row for row in rows}
    output = []
    for class_name in unknown_classes:
        item = {"Unknown Classes": class_name}
        for scale, label in (("s", "YW-Small"), ("m", "YW-Medium"), ("l", "YW-Large")):
            item[label] = per_class_map50(by_key.get(("yoloworld", "known_class", scale), {}), class_name)
        for label in ("SY-Small", "SY-Medium", "SY-Large"):
            item[label] = "0.0000"
        output.append(item)
    return output


def efficiency_row(row: dict[str, Any]) -> dict[str, str]:
    summary = row.get("research_summary") or {}
    all_metrics = (summary.get("metrics") or {}).get("all_class") or {}
    metric_summary = all_metrics.get("summary") or {}
    speed = metric_summary.get("speed") or {}
    inference = speed.get("inference") if isinstance(speed, dict) else None
    params = all_metrics.get("parameters")
    return {
        "Model & Training Config": MODEL_LABEL.get(row.get("model_type"), row.get("model_type", "")),
        "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
        "Parameters (M)": fmt((float(params) / 1_000_000.0) if params else None),
        "Inference Time (ms)": fmt(inference),
    }


def build_table5(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("model_type")), str(row.get("scale")))
        if row.get("training_config") == "all_class" or key not in best:
            best[key] = row
    return [efficiency_row(best[key]) for key in sorted(best, key=lambda k: ({"yoloworld": 0, "yolo": 1}.get(k[0], 9), {"s": 0, "m": 1, "l": 2}.get(k[1], 9)))]


def build_table6_pretrained_yoloworld(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(rows, key=model_sort):
        if row.get("model_type") != "yoloworld" or row.get("training_config") != "pretrained":
            continue
        item = {
            "Baseline": "YOLO-World Pretrained (No Fine-Tuning)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
            "Evaluation Target": "All Classes",
        }
        item.update(metric_row(row, "all"))
        output.append(item)

        unknown_item = {
            "Baseline": "YOLO-World Pretrained (No Fine-Tuning)",
            "Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
            "Evaluation Target": "Unknown Classes",
        }
        unknown_item.update(metric_row(row, "unknown"))
        output.append(unknown_item)
    return output


def table4_placeholder(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    output = []
    for row in sorted(rows, key=scale_sort):
        if row.get("model_type") == "yolo" and row.get("training_config") == "known_class":
            output.append({
                "Standard YOLO Scale": SCALE_LABEL.get(row.get("scale"), row.get("scale", "")),
                "Class-Agnostic Recall": "run class_agnostic_analysis script / pending",
                "Misclassification to Known Classes": "pending",
                "Miss Rate (False Negative Rate)": "pending",
            })
    return output


def write_all_tables(out_dir: Path, rows: list[dict[str, Any]], unknown_classes: list[str]) -> None:
    table_specs = [
        ("table1_all_class", build_table1(rows), ["Scheme ID", "Model & Training Config", "Scale", "$mAP_{50}$", "$mAP_{50-95}$", "mAP", "mIoU", "Precision", "Recall", "F1-Score"]),
        ("table2_zero_shot_known_train", build_table2(rows), ["Scheme ID", "Model & Training Config", "Scale", "Evaluation Target", "$mAP_{50}$", "$mAP_{50-95}$", "mAP", "mIoU", "Precision", "Recall", "F1-Score"]),
        ("table3_unknown_per_class_map50", build_table3(rows, unknown_classes), ["Unknown Classes", "YW-Small", "YW-Medium", "YW-Large", "SY-Small", "SY-Medium", "SY-Large"]),
        ("table4_standard_yolo_known_analysis", table4_placeholder(rows), ["Standard YOLO Scale", "Class-Agnostic Recall", "Misclassification to Known Classes", "Miss Rate (False Negative Rate)"]),
        ("table5_efficiency_latency", build_table5(rows), ["Model & Training Config", "Scale", "Parameters (M)", "Inference Time (ms)"]),
        ("table6_pretrained_yoloworld", build_table6_pretrained_yoloworld(rows), ["Baseline", "Scale", "Evaluation Target", "$mAP_{50}$", "$mAP_{50-95}$", "mAP", "mIoU", "Precision", "Recall", "F1-Score"]),
    ]
    for name, table_rows, headers in table_specs:
        write_csv(out_dir / f"{name}.csv", table_rows, headers)
        write_md_table(out_dir / f"{name}.md", table_rows, headers)


def write_report(out_dir: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = ["# Results Evaluation Report", "", f"Evaluated runs: {len(rows)}", f"Failures: {len(failures)}", "", "## Outputs", ""]
    for name in ("table1_all_class", "table2_zero_shot_known_train", "table3_unknown_per_class_map50", "table4_standard_yolo_known_analysis", "table5_efficiency_latency", "table6_pretrained_yoloworld"):
        lines.append(f"- `{name}.csv` / `{name}.md`")
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure.get('run_dir')}`: {failure.get('error')}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_file_logger(out_dir / "results_eval.log")
    run_dirs = discover_run_dirs([Path(root) for root in args.runs_root])
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result, failure = load_or_eval_run(run_dir, args, logger)
        if failure:
            failures.append(failure)
            continue
        if result:
            rows.append(summarize_run(result))
    rows.sort(key=model_sort)
    save_summary_csv(out_dir / "research_metrics_summary.csv", rows)
    save_json(out_dir / "research_metrics_summary.json", {"rows": rows, "failures": failures})
    write_all_tables(out_dir, rows, parse_csv(args.unknown_classes))
    write_report(out_dir, rows, failures)
    print(f"Discovered runs: {len(run_dirs)}")
    print(f"Evaluated/compiled: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Output: {out_dir}")
    if failures and not args.allow_failures:
        raise RuntimeError(f"{len(failures)} run(s) failed. See {out_dir / 'research_metrics_summary.json'}")


if __name__ == "__main__":
    main()
