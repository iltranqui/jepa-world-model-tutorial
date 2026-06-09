#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/helldiver/.virtualenvs/papers}"
CONFIG="${CONFIG:-configs/vjepa_lego_gears_yolo_finetune.yaml}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  echo "Create it and install torch, torchvision, pillow, pyyaml, and tqdm before running this script." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

mkdir -p runs
extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

python experiments/vjepa_lego_gears_run.py detect --config "$CONFIG" "${extra_args[@]}"
