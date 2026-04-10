#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODAL_BIN="modal"
MODE="full"
PIPELINES="text,image,multimodal"
NUM_EPOCHS=""
TEXT_BATCH_SIZE="16"
QWEN_BATCH_SIZE="1"
FREEZE_ENCODER="1"
DETACH="0"
SAVE_ROOT=""

usage() {
  cat <<'EOF'
Run the training pipelines on Modal with automatic resume.

Usage:
  bash training/run_all_pipelines_modal.sh [options]

Options:
  --modal-bin <path>              Modal CLI executable. Default: modal
  --smoke                         Use 1 epoch smoke mode
  --full                          Use full mode (default)
  --pipelines <csv>               Comma-separated subset: text,image,multimodal
  --num-epochs <n>                Total target epochs for each selected run
  --text-batch-size <n>           Text pipeline batch size. Default: 16
  --qwen-batch-size <n>           Qwen pipeline batch size. Default: 1
  --no-freeze-encoder             Disable --freeze_encoder for Qwen runs
  --save-root <dir>               Override remote checkpoint root
  --detach                        Run Modal job detached
  -h, --help                      Show this help

Behavior:
  - This launches training/modal_app.py::train on Modal.
  - Resume is handled remotely by training/run_all_pipelines.sh.
  - If latest.pt exists in the Modal checkpoint volume, the remote run resumes.
  - Otherwise it starts fresh.

Examples:
  bash training/run_all_pipelines_modal.sh
  bash training/run_all_pipelines_modal.sh --smoke
  bash training/run_all_pipelines_modal.sh --pipelines text
  bash training/run_all_pipelines_modal.sh --detach
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modal-bin)
      MODAL_BIN="$2"
      shift 2
      ;;
    --smoke)
      MODE="smoke"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --pipelines)
      PIPELINES="$2"
      shift 2
      ;;
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --text-batch-size)
      TEXT_BATCH_SIZE="$2"
      shift 2
      ;;
    --qwen-batch-size)
      QWEN_BATCH_SIZE="$2"
      shift 2
      ;;
    --no-freeze-encoder)
      FREEZE_ENCODER="0"
      shift
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --detach)
      DETACH="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cmd=(
  "$MODAL_BIN"
  "run"
  "training/modal_app.py::train"
  "--mode" "$MODE"
  "--pipelines" "$PIPELINES"
  "--text-batch-size" "$TEXT_BATCH_SIZE"
  "--qwen-batch-size" "$QWEN_BATCH_SIZE"
)

if [[ "$DETACH" == "1" ]]; then
  cmd=("$(printf '%s' "$MODAL_BIN")" "run" "-d" "training/modal_app.py::train" "--mode" "$MODE" "--pipelines" "$PIPELINES" "--text-batch-size" "$TEXT_BATCH_SIZE" "--qwen-batch-size" "$QWEN_BATCH_SIZE")
fi

if [[ -n "$NUM_EPOCHS" ]]; then
  cmd+=("--num-epochs" "$NUM_EPOCHS")
fi

if [[ "$FREEZE_ENCODER" == "0" ]]; then
  cmd+=("--no-freeze-encoder")
fi

if [[ -n "$SAVE_ROOT" ]]; then
  cmd+=("--save-root" "$SAVE_ROOT")
fi

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
