#!/usr/bin/env bash
set -euo pipefail

TRAINING_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_BUILD_ROOT=""
PRETRAINED_PATH="${AIC_STAGE_PRETRAINED_PATH:-}"
RUN_PREFIX="${AIC_STAGE_RUN_PREFIX:-sc_stage_fixed_scene_curated_001}"
DEFAULT_STEPS="${AIC_STAGE_TRAIN_STEPS:-10000}"
RETRY_STEPS="${AIC_STAGE_RETRY_STEPS:-10000}"
SAVE_FREQ="${AIC_STAGE_SAVE_FREQ:-2000}"
BATCH_SIZE="${AIC_STAGE_BATCH_SIZE:-8}"
NUM_WORKERS="${AIC_STAGE_NUM_WORKERS:-2}"
WANDB_FLAG="${AIC_STAGE_WANDB:-false}"
STAGES=()

while (($# > 0)); do
  case "$1" in
    --dataset-build-root)
      DATASET_BUILD_ROOT="$2"
      shift 2
      ;;
    --pretrained-path)
      PRETRAINED_PATH="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --stage)
      STAGES+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash quantarune_training/bash/train_sc_stage_teachers.sh \
    --dataset-build-root /path/to/act_stage_datasets/<run_root> \
    --pretrained-path /path/to/pretrained_model \
    --run-prefix sc_stage_fixed_scene_curated_001

Optional:
  --stage approach
  --stage align
  --stage guarded_insert
  --stage retry_or_finish

Environment overrides:
  AIC_STAGE_TRAIN_STEPS
  AIC_STAGE_RETRY_STEPS
  AIC_STAGE_SAVE_FREQ
  AIC_STAGE_BATCH_SIZE
  AIC_STAGE_NUM_WORKERS
  AIC_STAGE_WANDB
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${DATASET_BUILD_ROOT}" ]]; then
  echo "--dataset-build-root is required." >&2
  exit 2
fi
if [[ -z "${PRETRAINED_PATH}" ]]; then
  echo "--pretrained-path is required." >&2
  exit 2
fi
if [[ ! -d "${PRETRAINED_PATH}" ]]; then
  echo "Pretrained path does not exist: ${PRETRAINED_PATH}" >&2
  exit 1
fi

if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(approach align guarded_insert retry_or_finish)
fi

for stage in "${STAGES[@]}"; do
  dataset_root="${DATASET_BUILD_ROOT}/datasets/${stage}"
  if [[ ! -f "${dataset_root}/meta/tasks.parquet" ]]; then
    echo "Missing dataset root for stage ${stage}: ${dataset_root}" >&2
    exit 1
  fi

  stage_steps="${DEFAULT_STEPS}"
  if [[ "${stage}" == "retry_or_finish" ]]; then
    stage_steps="${RETRY_STEPS}"
  fi

  echo
  echo "=== Training stage ${stage} ==="
  echo "dataset_root=${dataset_root}"
  echo "pretrained_path=${PRETRAINED_PATH}"
  echo "steps=${stage_steps}"

  AIC_ACT_TRAIN_PRETRAINED_PATH="${PRETRAINED_PATH}" \
  AIC_ACT_TRAIN_STEPS="${stage_steps}" \
  AIC_ACT_TRAIN_BATCH_SIZE="${BATCH_SIZE}" \
  AIC_ACT_TRAIN_NUM_WORKERS="${NUM_WORKERS}" \
  AIC_ACT_TRAIN_SAVE_FREQ="${SAVE_FREQ}" \
  AIC_ACT_TRAIN_WANDB="${WANDB_FLAG}" \
  bash "${TRAINING_ROOT}/bash/train_act_dataset.sh" \
    --run-name "${RUN_PREFIX}_${stage}" \
    --repo-id "local/${RUN_PREFIX}-${stage}" \
    --dataset-root "${dataset_root}" \
    --no-publish
done
