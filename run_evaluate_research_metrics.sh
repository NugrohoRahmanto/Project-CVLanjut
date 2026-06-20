#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" scripts/evaluate_research_metrics.py \
  --runs-root runs/yolo_bdd10k \
  --runs-root runs/yoloworld_bdd10k \
  --data-yaml data/bdd10k/bdd10k.yaml \
  --summary-out runs/research_metrics_summary.csv \
  "$@"
