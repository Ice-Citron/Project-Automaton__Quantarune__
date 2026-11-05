#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${TRAINING_ROOT}/.pixi/envs/default/bin/python"
RUN_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"

RUN_NAME=""
REPO_ID=""
DATASET_ROOT=""
OUTPUT_DIR=""
PRETRAINED_PATH="${AIC_ACT_TRAIN_PRETRAINED_PATH:-}"
PRIVATE_FLAG="${AIC_ACT_TRAIN_PRIVATE:-true}"
PUBLISH_FLAG="${AIC_ACT_TRAIN_PUBLISH:-true}"

if [[ -f "${TRAINING_ROOT}/.secrets.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${TRAINING_ROOT}/.secrets.env"
  set +a
fi

while (($# > 0)); do
  case "$1" in
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --pretrained-path)
      PRETRAINED_PATH="$2"
      shift 2
      ;;
    --private)
      PRIVATE_FLAG=true
      shift
      ;;
    --no-private)
      PRIVATE_FLAG=false
      shift
      ;;
    --publish)
      PUBLISH_FLAG=true
      shift
      ;;
    --no-publish)
      PUBLISH_FLAG=false
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash quantarune_training/bash/train_act_dataset.sh \
    --run-name sfp_clean_198 \
    --repo-id shng2025/aic-act-sfp-clean-198 \
    --dataset-root /path/to/built_act_dataset

Optional flags:
  --output-dir /custom/output/path
  --pretrained-path /path/to/pretrained_model
  --private / --no-private
  --publish / --no-publish

Environment overrides:
  AIC_ACT_TRAIN_PRETRAINED_PATH
  AIC_ACT_TRAIN_STEPS
  AIC_ACT_TRAIN_BATCH_SIZE
  AIC_ACT_TRAIN_NUM_WORKERS
  AIC_ACT_TRAIN_DEVICE
  AIC_ACT_TRAIN_SAVE_FREQ
  AIC_ACT_TRAIN_LOG_FREQ
  AIC_ACT_TRAIN_WANDB
  AIC_ACT_TRAIN_WANDB_PROJECT
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
if [[ -z "${REPO_ID}" ]]; then
  echo "--repo-id is required." >&2
  exit 2
fi
if [[ -z "${DATASET_ROOT}" ]]; then
  echo "--dataset-root is required." >&2
  exit 2
fi
if [[ ! -f "${DATASET_ROOT}/meta/tasks.parquet" ]]; then
  echo "Dataset root is missing meta/tasks.parquet: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -n "${PRETRAINED_PATH}" && ! -d "${PRETRAINED_PATH}" ]]; then
  echo "Pretrained path does not exist: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${TRAINING_ROOT}/outputs/${RUN_NAME}/${RUN_TIMESTAMP}"
fi
mkdir -p "$(dirname "${OUTPUT_DIR}")"

export AIC_ACT_TRAIN_STEPS="${AIC_ACT_TRAIN_STEPS:-20000}"
export AIC_ACT_TRAIN_BATCH_SIZE="${AIC_ACT_TRAIN_BATCH_SIZE:-8}"
export AIC_ACT_TRAIN_NUM_WORKERS="${AIC_ACT_TRAIN_NUM_WORKERS:-2}"
export AIC_ACT_TRAIN_DEVICE="${AIC_ACT_TRAIN_DEVICE:-cuda}"
export AIC_ACT_TRAIN_SAVE_FREQ="${AIC_ACT_TRAIN_SAVE_FREQ:-5000}"
export AIC_ACT_TRAIN_LOG_FREQ="${AIC_ACT_TRAIN_LOG_FREQ:-100}"
export AIC_ACT_TRAIN_WANDB="${AIC_ACT_TRAIN_WANDB:-true}"
export AIC_ACT_TRAIN_WANDB_PROJECT="${AIC_ACT_TRAIN_WANDB_PROJECT:-aic-act-training}"

TRAIN_ARGS=(
  --policy.type act
  --policy.push_to_hub false
  --dataset.repo_id "$(basename "${DATASET_ROOT}")"
  --dataset.root "${DATASET_ROOT}"
  --dataset.video_backend pyav
  --output_dir "${OUTPUT_DIR}"
  --job_name "${RUN_NAME}_${RUN_TIMESTAMP}"
  --policy.device "${AIC_ACT_TRAIN_DEVICE}"
  --batch_size "${AIC_ACT_TRAIN_BATCH_SIZE}"
  --steps "${AIC_ACT_TRAIN_STEPS}"
  --save_freq "${AIC_ACT_TRAIN_SAVE_FREQ}"
  --eval_freq 1000000
  --log_freq "${AIC_ACT_TRAIN_LOG_FREQ}"
  --num_workers "${AIC_ACT_TRAIN_NUM_WORKERS}"
)

if [[ -n "${PRETRAINED_PATH}" ]]; then
  TRAIN_ARGS+=(--policy.pretrained_path "${PRETRAINED_PATH}")
fi

if [[ "${AIC_ACT_TRAIN_WANDB}" == "true" ]]; then
  TRAIN_ARGS+=(--wandb.enable true --wandb.project "${AIC_ACT_TRAIN_WANDB_PROJECT}")
else
  TRAIN_ARGS+=(--wandb.enable false)
fi

echo "ACT training settings:"
echo "  RUN_NAME=${RUN_NAME}"
echo "  REPO_ID=${REPO_ID}"
echo "  DATASET_ROOT=${DATASET_ROOT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  PRETRAINED_PATH=${PRETRAINED_PATH:-<none>}"
echo "  AIC_ACT_TRAIN_STEPS=${AIC_ACT_TRAIN_STEPS}"
echo "  AIC_ACT_TRAIN_BATCH_SIZE=${AIC_ACT_TRAIN_BATCH_SIZE}"
echo "  AIC_ACT_TRAIN_NUM_WORKERS=${AIC_ACT_TRAIN_NUM_WORKERS}"
echo "  AIC_ACT_TRAIN_DEVICE=${AIC_ACT_TRAIN_DEVICE}"
echo "  AIC_ACT_TRAIN_WANDB=${AIC_ACT_TRAIN_WANDB}"
echo "  PRIVATE_FLAG=${PRIVATE_FLAG}"
echo "  PUBLISH_FLAG=${PUBLISH_FLAG}"

"${PYTHON_BIN}" "${TRAINING_ROOT}/scripts/lerobot_train_patched.py" "${TRAIN_ARGS[@]}"

if [[ "${PUBLISH_FLAG}" == "true" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN not set; cannot publish ${RUN_NAME}." >&2
    exit 1
  fi

  PUBLISH_ARGS=(
    --output-dir "${OUTPUT_DIR}"
    --repo-id "${REPO_ID}"
    --branch-prefix "${RUN_NAME}"
    --dataset-label "${DATASET_ROOT}"
    --device "${AIC_ACT_TRAIN_DEVICE}"
    --upload-main
    --upload-checkpoints
  )
  if [[ "${PRIVATE_FLAG}" == "true" ]]; then
    PUBLISH_ARGS+=(--private)
  else
    PUBLISH_ARGS+=(--no-private)
  fi

  "${PYTHON_BIN}" "${TRAINING_ROOT}/scripts/publish_act_run.py" "${PUBLISH_ARGS[@]}"
fi

echo
echo "Completed ACT training run:"
echo "  DATASET_ROOT=${DATASET_ROOT}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
