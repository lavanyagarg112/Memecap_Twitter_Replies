#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="python3"
DEVICE="auto"
MODE="full"
SAVE_ROOT="training/checkpoints"
IMAGE_DIR="training/data/non-annotation-dataset/images"
TEXT_BATCH_SIZE="16"
QWEN_BATCH_SIZE="1"
NUM_EPOCHS=""
FREEZE_ENCODER="1"
ONLY_PIPELINES="text,image,multimodal"

usage() {
  cat <<'EOF'
Run the three main training pipelines sequentially with automatic resume.

Usage:
  bash training/run_all_pipelines.sh [options]

Options:
  --device <auto|cuda|cuda:0|mps|cpu>   Training device. Default: auto
  --python <python_bin>                 Python executable. Default: python3
  --smoke                               Use 1 epoch and *_smoke checkpoint dirs
  --full                                Use normal checkpoint dirs (default)
  --num-epochs <n>                      Total target epochs for each run
  --save-root <dir>                     Parent checkpoint dir
  --image-dir <dir>                     Downloaded image directory
  --text-batch-size <n>                 Text pipeline batch size. Default: 16
  --qwen-batch-size <n>                 Qwen pipeline batch size. Default: 1
  --no-freeze-encoder                   Do not pass --freeze_encoder to Qwen runs
  --only <csv>                          Subset of pipelines: text,image,multimodal
  -h, --help                            Show this help

Behavior:
  - If <save_dir>/latest.pt exists, the script adds --resume.
  - If not, the script starts a fresh run for that pipeline.
  - If the checkpoint already reached the requested epoch count, train.py skips it.

Examples:
  bash training/run_all_pipelines.sh
  bash training/run_all_pipelines.sh --smoke
  bash training/run_all_pipelines.sh --device cuda --num-epochs 20
  bash training/run_all_pipelines.sh --only text
EOF
}

contains_pipeline() {
  local target="$1"
  local item
  IFS=',' read -r -a items <<<"$ONLY_PIPELINES"
  for item in "${items[@]}"; do
    if [[ "$item" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

validate_pipelines() {
  local item
  local count=0
  IFS=',' read -r -a items <<<"$ONLY_PIPELINES"
  for item in "${items[@]}"; do
    case "$item" in
      text|image|multimodal)
        count=$((count + 1))
        ;;
      *)
        echo "Invalid pipeline in --only: $item" >&2
        echo "Allowed values: text,image,multimodal" >&2
        exit 1
        ;;
    esac
  done
  if [[ "$count" -eq 0 ]]; then
    echo "No pipelines selected." >&2
    exit 1
  fi
}

detect_device() {
  if [[ "$DEVICE" != "auto" ]]; then
    printf '%s\n' "$DEVICE"
    return
  fi

  "$PYTHON_BIN" - <<'PY'
import torch

if torch.cuda.is_available():
    print("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
}

run_pipeline() {
  local label="$1"
  local save_dir="$2"
  shift 2

  local cmd=("$PYTHON_BIN" "training/train.py" "$@" "--device" "$DEVICE_RESOLVED" "--save_dir" "$save_dir")

  if [[ -n "$NUM_EPOCHS" ]]; then
    cmd+=("--num_epochs" "$NUM_EPOCHS")
  fi

  if [[ -f "$save_dir/latest.pt" ]]; then
    cmd+=("--resume")
  fi

  printf '\n[%s]\n' "$label"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
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
    --num-epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --image-dir)
      IMAGE_DIR="$2"
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
    --only)
      ONLY_PIPELINES="$2"
      shift 2
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

validate_pipelines
DEVICE_RESOLVED="$(detect_device)"

if [[ "$MODE" == "smoke" ]]; then
  : "${NUM_EPOCHS:=1}"
  TEXT_SAVE_DIR="$SAVE_ROOT/text_hf_smoke"
  IMAGE_SAVE_DIR="$SAVE_ROOT/image_qwen_smoke"
  MULTIMODAL_SAVE_DIR="$SAVE_ROOT/multimodal_qwen_smoke"
else
  TEXT_SAVE_DIR="$SAVE_ROOT/text_hf_clean"
  IMAGE_SAVE_DIR="$SAVE_ROOT/image_qwen_clean"
  MULTIMODAL_SAVE_DIR="$SAVE_ROOT/multimodal_qwen_clean"
fi

if [[ "$DEVICE_RESOLVED" == "cpu" ]]; then
  echo "Warning: auto-selected device=cpu. The Qwen image/multimodal runs may be extremely slow." >&2
fi

if [[ "$DEVICE_RESOLVED" == "mps" ]]; then
  echo "Warning: device=mps. The Qwen image/multimodal runs may not be as stable as CUDA." >&2
fi

if { contains_pipeline "image" || contains_pipeline "multimodal"; } && [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Image directory not found: $IMAGE_DIR" >&2
  echo "Download images first with: python3 training/download_images.py" >&2
  exit 1
fi

echo "Root directory: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Device: $DEVICE_RESOLVED"
echo "Mode: $MODE"
echo "Pipelines: $ONLY_PIPELINES"

if contains_pipeline "text"; then
  run_pipeline \
    "text" \
    "$TEXT_SAVE_DIR" \
    --pipeline text \
    --encoder_type hf \
    --batch_size "$TEXT_BATCH_SIZE"
fi

qwen_args=()
if [[ "$FREEZE_ENCODER" == "1" ]]; then
  qwen_args+=(--freeze_encoder)
fi

if contains_pipeline "image"; then
  run_pipeline \
    "image" \
    "$IMAGE_SAVE_DIR" \
    --pipeline image \
    --encoder_type qwen_vl \
    --batch_size "$QWEN_BATCH_SIZE" \
    --image_dir "$IMAGE_DIR" \
    "${qwen_args[@]}"
fi

if contains_pipeline "multimodal"; then
  run_pipeline \
    "multimodal" \
    "$MULTIMODAL_SAVE_DIR" \
    --pipeline multimodal \
    --encoder_type qwen_vl \
    --batch_size "$QWEN_BATCH_SIZE" \
    --image_dir "$IMAGE_DIR" \
    "${qwen_args[@]}"
fi
