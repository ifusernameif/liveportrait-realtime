#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${LIVEPORTRAIT_ENV_NAME:-LivePortrait}"
SOURCE=""
CAMERA="0"
TARGET_FPS="15"
DISPLAY="generated"
MIRROR_INPUT=0
MIRROR_OUTPUT=0
FAST=1
USE_APP=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_realtime_macos.sh --source /path/to/portrait.jpg [options]

Options:
  --source PATH         Source portrait image. Required unless ./personal.jpg exists.
  --camera VALUE        Camera index or stream URL. Default: 0.
  --target-fps VALUE    Target processing FPS. Try 12, 15, 18, 24. Default: 15.
  --display VALUE       generated or split. Default: generated.
  --mirror-input        Mirror camera input before driving.
  --mirror-output       Mirror generated output for preview/live use.
  --no-fast             Use the default crop settings instead of the low-latency preset.
  --app                 Create and open the macOS .app launcher.
  --                    Pass remaining arguments to realtime_camera.py.

Examples:
  bash scripts/run_realtime_macos.sh --source personal.jpg --camera 0 --target-fps 15 --mirror-output
  bash scripts/run_realtime_macos.sh --source personal.jpg --camera 0 --app
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"
      shift 2
      ;;
    --camera)
      CAMERA="${2:-0}"
      shift 2
      ;;
    --target-fps)
      TARGET_FPS="${2:-15}"
      shift 2
      ;;
    --display)
      DISPLAY="${2:-generated}"
      shift 2
      ;;
    --mirror-input)
      MIRROR_INPUT=1
      shift
      ;;
    --mirror-output)
      MIRROR_OUTPUT=1
      shift
      ;;
    --no-fast)
      FAST=0
      shift
      ;;
    --app)
      USE_APP=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$SOURCE" && -f personal.jpg ]]; then
  SOURCE="personal.jpg"
fi
if [[ -z "$SOURCE" ]]; then
  usage
  echo
  echo "Error: --source is required." >&2
  exit 2
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "Error: source image not found: $SOURCE" >&2
  exit 2
fi

find_conda() {
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in \
    /opt/homebrew/bin/conda \
    /opt/homebrew/Caskroom/miniforge/base/bin/conda \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! CONDA_BIN="$(find_conda)"; then
  if command -v brew >/dev/null 2>&1; then
    echo "conda not found; installing Miniforge with Homebrew..."
    brew install --cask miniforge
    CONDA_BIN="$(find_conda)"
  else
    echo "Error: conda not found. Install Miniforge first: https://github.com/conda-forge/miniforge" >&2
    exit 1
  fi
fi

CONDA_BASE="$("$CONDA_BIN" info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda env: $ENV_NAME"
  conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "Installing/checking Python dependencies..."
if ! conda run -n "$ENV_NAME" python - <<'PY' >/dev/null 2>&1
import cv2, torch, requests, tyro
PY
then
  conda run -n "$ENV_NAME" python -m pip install -r requirements_macOS.txt
  conda run -n "$ENV_NAME" python -m pip install -U "huggingface_hub[cli]"
fi

missing_weights=0
for required in \
  pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth \
  pretrained_weights/liveportrait/base_models/motion_extractor.pth \
  pretrained_weights/liveportrait/base_models/spade_generator.pth \
  pretrained_weights/liveportrait/base_models/warping_module.pth \
  pretrained_weights/liveportrait/landmark.onnx \
  pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth \
  pretrained_weights/insightface/models/buffalo_l/det_10g.onnx \
  pretrained_weights/insightface/models/buffalo_l/2d106det.onnx; do
  [[ -e "$required" ]] || missing_weights=1
done

if [[ "$missing_weights" -eq 1 ]]; then
  echo "Downloading LivePortrait pretrained weights..."
  conda run -n "$ENV_NAME" hf download KlingTeam/LivePortrait \
    --local-dir pretrained_weights \
    --exclude "*.git*" \
    --exclude "README.md" \
    --exclude "docs/*"
fi

if [[ "$USE_APP" -eq 1 ]]; then
  app_args=(tools/create_macos_camera_app.py --source "$SOURCE" --camera "$CAMERA" --display "$DISPLAY" --target-fps "$TARGET_FPS")
  [[ "$FAST" -eq 1 ]] && app_args+=(--fast)
  [[ "$MIRROR_INPUT" -eq 1 ]] && app_args+=(--mirror-input)
  [[ "$MIRROR_OUTPUT" -eq 1 ]] && app_args+=(--mirror-output)
  conda run -n "$ENV_NAME" python "${app_args[@]}"
  xattr -dr com.apple.quarantine dist/LivePortraitCamera.app 2>/dev/null || true
  open dist/LivePortraitCamera.app
  echo "Opened dist/LivePortraitCamera.app. Logs: logs/LivePortraitCamera.log"
  exit 0
fi

run_args=(
  realtime_camera.py
  --source "$SOURCE"
  --camera "$CAMERA"
  --display "$DISPLAY"
  --camera-fps "$TARGET_FPS"
  --target-fps "$TARGET_FPS"
)
if [[ "$FAST" -eq 1 ]]; then
  run_args+=(
    --camera-width 640
    --camera-height 480
    --driving-crop-mode static
    --redetect-interval 240
    --motion-smoothing 0.15
  )
fi
[[ "$MIRROR_INPUT" -eq 1 ]] && run_args+=(--mirror-input)
[[ "$MIRROR_OUTPUT" -eq 1 ]] && run_args+=(--mirror-output)
run_args+=("${EXTRA_ARGS[@]}")

echo "Starting LivePortrait realtime..."
conda run -n "$ENV_NAME" python "${run_args[@]}"
