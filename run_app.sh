#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
STREAMLIT_BIN="${STREAMLIT_BIN:-.venv/bin/streamlit}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

if [[ ! -x "$STREAMLIT_BIN" ]]; then
  if [[ -x "$PYTHON_BIN" ]]; then
    STREAMLIT_BIN="$PYTHON_BIN -m streamlit"
  else
    STREAMLIT_BIN="python -m streamlit"
  fi
fi

echo "Starting Streamlit inference app"
echo "Host: $HOST"
echo "Port: $PORT"
echo "App : app/app.py"
echo "Console will show model loading, prompt setup, inference start, and bbox summaries."

exec $STREAMLIT_BIN run app/app.py \
  --server.address "$HOST" \
  --server.port "$PORT" \
  --logger.level info
