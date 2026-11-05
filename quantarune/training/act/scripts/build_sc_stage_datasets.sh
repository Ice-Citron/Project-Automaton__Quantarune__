#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${TRAINING_ROOT}/.." && pwd)"
TELEOP_ROOT="${REPO_ROOT}/quantarune_teleop"
PYTHON_BIN="${TRAINING_ROOT}/.pixi/envs/default/bin/python"
RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

RUN_NAME=""
OUTPUT_ROOT=""
TELEOP_OUTPUT_ROOT="${AIC_SC_STAGE_TELEOP_ROOT:-${TELEOP_ROOT}/outputs/gazebo_teleop}"
SESSION_PREFIX="${AIC_SC_STAGE_SESSION_PREFIX:-sc_}"
SKIP_TELEOP_ROOT="${AIC_SC_STAGE_SKIP_TELEOP_ROOT:-false}"
MIN_SEGMENT_FRAMES="${AIC_SC_STAGE_MIN_SEGMENT_FRAMES:-8}"
IMAGE_SCALE="${AIC_SC_STAGE_IMAGE_SCALE:-0.25}"
IMAGE_WRITER_THREADS="${AIC_SC_STAGE_IMAGE_WRITER_THREADS:-10}"
IMAGE_WRITER_PROCESSES="${AIC_SC_STAGE_IMAGE_WRITER_PROCESSES:-0}"
BATCH_ENCODING_SIZE="${AIC_SC_STAGE_BATCH_ENCODING_SIZE:-1}"
VCODEC="${AIC_SC_STAGE_VCODEC:-h264}"
INCLUDE_SESSIONS=()
EXCLUDE_SESSIONS=()
CASES_CSVS=()

while (($# > 0)); do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --teleop-root)
      TELEOP_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --session-prefix)
      SESSION_PREFIX="$2"
      shift 2
      ;;
    --skip-teleop-root)
      SKIP_TELEOP_ROOT=true
      shift
      ;;
    --include-session)
      INCLUDE_SESSIONS+=("$2")
      shift 2
      ;;
    --exclude-session)
      EXCLUDE_SESSIONS+=("$2")
      shift 2
      ;;
    --cases-csv)
      CASES_CSVS+=("$2")
      shift 2
      ;;
    --min-segment-frames)
      MIN_SEGMENT_FRAMES="$2"
      shift 2
      ;;
    --image-scale)
      IMAGE_SCALE="$2"
      shift 2
      ;;
    --image-writer-threads)
      IMAGE_WRITER_THREADS="$2"
      shift 2
      ;;
    --image-writer-processes)
      IMAGE_WRITER_PROCESSES="$2"
      shift 2
      ;;
    --batch-encoding-size)
      BATCH_ENCODING_SIZE="$2"
      shift 2
      ;;
    --vcodec)
      VCODEC="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash quantarune_training/bash/build_sc_stage_datasets.sh \
    --run-name sc_stage_teleop_v1

Optional flags:
  --output-root /custom/output/path
  --teleop-root /path/to/quantarune_teleop/outputs/gazebo_teleop
  --session-prefix sc_
  --skip-teleop-root
  --cases-csv /path/to/filtered_cases.csv
  --include-session sc_intervention_fullboard_001
  --exclude-session sc_manual_batch_20260426_001
  --min-segment-frames 8
  --image-scale 0.25
  --image-writer-threads 10
  --image-writer-processes 0
  --batch-encoding-size 1
  --vcodec h264

Environment overrides:
  AIC_SC_STAGE_TELEOP_ROOT
  AIC_SC_STAGE_SESSION_PREFIX
  AIC_SC_STAGE_SKIP_TELEOP_ROOT
  AIC_SC_STAGE_MIN_SEGMENT_FRAMES
  AIC_SC_STAGE_IMAGE_SCALE
  AIC_SC_STAGE_IMAGE_WRITER_THREADS
  AIC_SC_STAGE_IMAGE_WRITER_PROCESSES
  AIC_SC_STAGE_BATCH_ENCODING_SIZE
  AIC_SC_STAGE_VCODEC
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RUN_NAME}" ]]; then
  echo "--run-name is required." >&2
  exit 2
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${TRAINING_ROOT}/outputs/act_stage_datasets/${RUN_NAME}_${RUN_TIMESTAMP}"
fi

MANIFEST_ROOT="${OUTPUT_ROOT}/stage_supervision"
DATASETS_ROOT="${OUTPUT_ROOT}/datasets"

echo "SC stage dataset build settings:"
echo "  RUN_NAME=${RUN_NAME}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  TELEOP_OUTPUT_ROOT=${TELEOP_OUTPUT_ROOT}"
echo "  SESSION_PREFIX=${SESSION_PREFIX}"
echo "  SKIP_TELEOP_ROOT=${SKIP_TELEOP_ROOT}"
echo "  MIN_SEGMENT_FRAMES=${MIN_SEGMENT_FRAMES}"
echo "  IMAGE_SCALE=${IMAGE_SCALE}"
echo "  IMAGE_WRITER_THREADS=${IMAGE_WRITER_THREADS}"
echo "  IMAGE_WRITER_PROCESSES=${IMAGE_WRITER_PROCESSES}"
echo "  BATCH_ENCODING_SIZE=${BATCH_ENCODING_SIZE}"
echo "  VCODEC=${VCODEC}"
if [[ ${#INCLUDE_SESSIONS[@]} -gt 0 ]]; then
  printf '  INCLUDE_SESSIONS=\n'
  for session in "${INCLUDE_SESSIONS[@]}"; do
    printf '    %s\n' "${session}"
  done
fi
if [[ ${#EXCLUDE_SESSIONS[@]} -gt 0 ]]; then
  printf '  EXCLUDE_SESSIONS=\n'
  for session in "${EXCLUDE_SESSIONS[@]}"; do
    printf '    %s\n' "${session}"
  done
fi
if [[ ${#CASES_CSVS[@]} -gt 0 ]]; then
  printf '  CASES_CSVS=\n'
  for cases_csv in "${CASES_CSVS[@]}"; do
    printf '    %s\n' "${cases_csv}"
  done
fi

MANIFEST_ARGS=(
  "${TELEOP_ROOT}/scripts/build_sc_stage_manifest.py"
  --teleop-root "${TELEOP_OUTPUT_ROOT}"
  --output-root "${MANIFEST_ROOT}"
  --session-prefix "${SESSION_PREFIX}"
  --force
)
if [[ "${SKIP_TELEOP_ROOT}" == "true" ]]; then
  MANIFEST_ARGS+=(--skip-teleop-root)
fi
for session in "${INCLUDE_SESSIONS[@]}"; do
  MANIFEST_ARGS+=(--include-session "${session}")
done
for session in "${EXCLUDE_SESSIONS[@]}"; do
  MANIFEST_ARGS+=(--exclude-session "${session}")
done
for cases_csv in "${CASES_CSVS[@]}"; do
  MANIFEST_ARGS+=(--cases-csv "${cases_csv}")
done

"${PYTHON_BIN}" "${MANIFEST_ARGS[@]}"

BUILD_ARGS=(
  "${TRAINING_ROOT}/scripts/build_sc_stage_act_datasets.py"
  --stage-supervision-json "${MANIFEST_ROOT}/sc_fsm_stage_supervision.json"
  --output-root "${DATASETS_ROOT}"
  --min-segment-frames "${MIN_SEGMENT_FRAMES}"
  --image-scale "${IMAGE_SCALE}"
  --image-writer-threads "${IMAGE_WRITER_THREADS}"
  --image-writer-processes "${IMAGE_WRITER_PROCESSES}"
  --batch-encoding-size "${BATCH_ENCODING_SIZE}"
  --vcodec "${VCODEC}"
  --force
)

"${PYTHON_BIN}" "${BUILD_ARGS[@]}"

echo
echo "SC stage datasets ready:"
echo "  SUPERVISION_JSON=${MANIFEST_ROOT}/sc_fsm_stage_supervision.json"
echo "  FRAME_RANGES_CSV=${MANIFEST_ROOT}/sc_fsm_stage_frame_ranges.csv"
echo "  DATASETS_ROOT=${DATASETS_ROOT}"
