#!/usr/bin/env bash
set -euo pipefail

# Demo training script for the v2 architecture.
# Run from the repo root:
#   bash model/demo_train_v2.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"
OUT_DIR="${OUT_DIR:-/private/tmp/haemmr_v2_runs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-demo_v2_${STAMP}}"
CSV_PATH="$OUT_DIR/${NAME}.csv"
CKPT_PATH="$OUT_DIR/${NAME}.pt"

mkdir -p "$OUT_DIR"

exec "$VENV_PY" "$ROOT_DIR/model/train.py" \
  --dataset wikitext-2 \
  --data-root "$ROOT_DIR/model/.data" \
  --vocab-cap 512 \
  --max-train-tokens 500000 \
  --D 1024 \
  --layers 1 \
  --d-ff 2048 \
  --slots 64 \
  --top-k 5 \
  --epi-read-k 1 \
  --seq-len 64 \
  --batch-size 16 \
  --steps 4096 \
  --eta 3.0 \
  --threshold 8.0 \
  --eta-end 1.0 \
  --eval-every 100 \
  --sample-every 100 \
  --sample-len 40 \
  --temperature 0.8 \
  --sample-top-k 20 \
  --log-csv "$CSV_PATH" \
  --ckpt "$CKPT_PATH" \
  --device cpu
