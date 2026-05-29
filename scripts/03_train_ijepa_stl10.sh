#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/helldiver/.virtualenvs/papers}"
CONFIG="${CONFIG:-configs/ijepa_stl10_pretrain.yaml}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing virtualenv: $VENV_DIR" >&2
  echo "Create it and install torch, torchvision, and tqdm before running this script." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

mkdir -p datasets runs
extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

python experiments/ijepa_image_run.py pretrain --config "$CONFIG" "${extra_args[@]}"
