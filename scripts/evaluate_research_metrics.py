from __future__ import annotations

import argparse
from pathlib import Path

from research_evaluation import (
    configure_file_logger,
    discover_run_dirs,
    evaluate_run_dir,
    parse_csv,
    save_json,
    save_summary_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing YOLO/YOLO-World BDD10K checkpoints with research metrics: "
            "all-class GT and unknown-class GT."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run-dir", default="", help="Single finished experiment directory containing weights/best.pt.")
    target.add_argument(
        "--runs-root",
        action="append",
        default=[],
        help="Root directory with experiment subfolders. Can be provided multiple times.",
    )
    parser.add_argument("--data-yaml", default="data/bdd10k/bdd10k.yaml")
    parser.add_argument("--model-type", choices=("auto", "yolo", "yoloworld"), default="auto")
    parser.add_argument("--known-classes", default="car,bus,truck")
    parser.add_argument("--unknown-classes", default="")
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--output-name", default="research_eval")
    parser.add_argument("--summary-out", default="runs/research_metrics_summary.csv")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_yaml = Path(args.data_yaml)
    known_classes = parse_csv(args.known_classes)
    unknown_classes = parse_csv(args.unknown_classes)
    logger = configure_file_logger(Path("research_evaluation.log"))

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        run_dirs = discover_run_dirs([Path(root) for root in args.runs_root])
    if not run_dirs:
        raise RuntimeError("No experiment run directories with weights/best.pt or weights/last.pt were found.")

    rows = []
    failures = []
    for run_dir in run_dirs:
        try:
            result = evaluate_run_dir(
                run_dir=run_dir,
                source_data_yaml=data_yaml,
                output_name=args.output_name,
                model_type=args.model_type,
                known_classes=known_classes,
                unknown_classes=unknown_classes,
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
            rows.append(result["summary_row"])
        except Exception as exc:
            logger.exception("Research evaluation failed for run_dir=%s", run_dir)
            failures.append({"run_dir": str(run_dir), "error": str(exc)})

    save_summary_csv(Path(args.summary_out), rows)
    save_json(Path(args.summary_out).with_suffix(".json"), {"rows": rows, "failures": failures})
    print(f"Evaluated runs: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Summary CSV: {args.summary_out}")
    print("Log: research_evaluation.log")
    if failures:
        raise RuntimeError(f"{len(failures)} run(s) failed. See research_evaluation.log and {Path(args.summary_out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
