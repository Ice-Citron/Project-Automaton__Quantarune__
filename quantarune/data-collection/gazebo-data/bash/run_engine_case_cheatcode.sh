#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${DATA_ROOT}/.." && pwd)"
RESET_SCRIPT="${REPO_ROOT}/quantarune_teleop/bash/reset_gazebo_eval_stack.sh"

SESSION_NAME="${ENGINE_CASE_SESSION_NAME:-engine_case_$(date +%Y%m%d_%H%M%S)}"
SESSION_ROOT="${DATA_ROOT}/outputs/fixed_scene_runs/${SESSION_NAME}"
LOG_ROOT="${SESSION_ROOT}/logs"

CONFIG_FILE=""
ENGINE_WAIT_SECONDS="${ENGINE_CASE_ENGINE_WAIT_SECONDS:-16}"
POLICY_WAIT_SECONDS="${ENGINE_CASE_POLICY_WAIT_SECONDS:-4}"
RECORDER_READY_SECONDS="${ENGINE_CASE_RECORDER_READY_SECONDS:-6}"
RECORDER_FLUSH_SECONDS="${ENGINE_CASE_RECORDER_FLUSH_SECONDS:-8}"
AFTER_ENGINE_EXIT_GRACE_SECONDS="${ENGINE_CASE_AFTER_ENGINE_EXIT_GRACE_SECONDS:-30}"
RESULT_TIMEOUT_SECONDS="${ENGINE_CASE_RESULT_TIMEOUT_SECONDS:-260}"
AUTO_CLEAN_STALE="${ENGINE_CASE_AUTO_CLEAN_STALE:-1}"
CLEAN_SETTLE_SECONDS="${ENGINE_CASE_CLEAN_SETTLE_SECONDS:-6}"
RECORD_SESSION_NAME=""
RECORD_NUMBER_TRIALS="${ENGINE_CASE_RECORD_NUMBER_TRIALS:-1}"
POLICY_MODE="${ENGINE_CASE_POLICY_MODE:-full}"
POLICY_IMPORT="${ENGINE_CASE_POLICY_IMPORT:-aic_example_policies.ros.CheatCode}"
HANDOFF_Z_OFFSET_M="${ENGINE_CASE_HANDOFF_Z_OFFSET_M:-0.05}"
HANDOFF_SETTLE_S="${ENGINE_CASE_HANDOFF_SETTLE_S:-1.5}"
SCORING_OUTPUT_PATH="${ENGINE_CASE_SCORING_YAML:-${HOME}/aic_results/scoring.yaml}"
SCORING_WAIT_SECONDS="${ENGINE_CASE_SCORING_WAIT_SECONDS:-20}"
SCORING_EXTRACTOR="${DATA_ROOT}/scripts/extract_engine_scoring.py"

usage() {
  cat <<'EOF'
Usage:
  bash quantarune_data/bash/run_engine_case_cheatcode.sh [options]

Options:
  --config-file PATH       Single-trial aic_engine config file to run. Required.
  --session-name NAME      Session directory name under outputs/fixed_scene_runs/.
  --record-session NAME    Dataset session name under outputs/gazebo_data/. Defaults to session name.
  --record-trials N        Recorder auto-stop count. Default: 1.
  --policy-mode MODE       CheatCode mode: full, handoff, or auto_rescue. Default: full.
  --policy-import PATH     Python import path for the policy module. Default: aic_example_policies.ros.CheatCode.
  --handoff-z-offset-m M   Handoff stop height above the port when policy mode is handoff. Default: 0.05.
  --handoff-settle-s SEC   Handoff settle time when policy mode is handoff. Default: 1.5.
  --engine-wait SECONDS    Wait after engine bringup before starting recorder/policy. Default: 16.
  --policy-wait SECONDS    Wait after policy startup. Default: 4.
  --recorder-ready SECONDS Wait after recorder startup before policy launch. Default: 6.
  --result-timeout SECONDS Maximum wall time to wait for recorder completion. Default: 260.
  --scoring-wait SECONDS   Wait after recorder flush for fresh scoring.yaml. Default: 20.
  --no-clean               Skip reset_gazebo_eval_stack.sh before launch.
  -h, --help               Show help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="$2"
      SESSION_ROOT="${DATA_ROOT}/outputs/fixed_scene_runs/${SESSION_NAME}"
      LOG_ROOT="${SESSION_ROOT}/logs"
      shift 2
      ;;
    --record-session)
      RECORD_SESSION_NAME="$2"
      shift 2
      ;;
    --record-trials)
      RECORD_NUMBER_TRIALS="$2"
      shift 2
      ;;
    --policy-mode)
      POLICY_MODE="$2"
      shift 2
      ;;
    --policy-import)
      POLICY_IMPORT="$2"
      shift 2
      ;;
    --handoff-z-offset-m)
      HANDOFF_Z_OFFSET_M="$2"
      shift 2
      ;;
    --handoff-settle-s)
      HANDOFF_SETTLE_S="$2"
      shift 2
      ;;
    --engine-wait)
      ENGINE_WAIT_SECONDS="$2"
      shift 2
      ;;
    --policy-wait)
      POLICY_WAIT_SECONDS="$2"
      shift 2
      ;;
    --recorder-ready)
      RECORDER_READY_SECONDS="$2"
      shift 2
      ;;
    --result-timeout)
      RESULT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --scoring-wait)
      SCORING_WAIT_SECONDS="$2"
      shift 2
      ;;
    --no-clean)
      AUTO_CLEAN_STALE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "--config-file is required" >&2
  exit 2
fi

CONFIG_FILE="$(realpath "${CONFIG_FILE}")"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

ENGINE_LOG="${LOG_ROOT}/engine.log"
POLICY_LOG="${LOG_ROOT}/policy.log"
RECORDER_LOG="${LOG_ROOT}/recorder.log"
MANIFEST_PATH="${SESSION_ROOT}/manifest.txt"
STAGE_TRACE_PATH="${SESSION_ROOT}/teacher_stage_trace.json"

ENGINE_PID=""
POLICY_PID=""
RECORDER_PID=""

write_manifest() {
  cat > "${MANIFEST_PATH}" <<EOF
session_name=${SESSION_NAME}
session_root=${SESSION_ROOT}
config_file=${CONFIG_FILE}
engine_log=${ENGINE_LOG}
policy_log=${POLICY_LOG}
recorder_log=${RECORDER_LOG}
engine_wait_seconds=${ENGINE_WAIT_SECONDS}
policy_wait_seconds=${POLICY_WAIT_SECONDS}
recorder_ready_seconds=${RECORDER_READY_SECONDS}
after_engine_exit_grace_seconds=${AFTER_ENGINE_EXIT_GRACE_SECONDS}
result_timeout_seconds=${RESULT_TIMEOUT_SECONDS}
policy_mode=${POLICY_MODE}
policy_import=${POLICY_IMPORT}
teacher_stage_trace=${STAGE_TRACE_PATH}
handoff_z_offset_m=${HANDOFF_Z_OFFSET_M}
handoff_settle_s=${HANDOFF_SETTLE_S}
scoring_output_path=${SCORING_OUTPUT_PATH}
scoring_wait_seconds=${SCORING_WAIT_SECONDS}
record_session_name=${RECORD_SESSION_NAME}
record_number_trials=${RECORD_NUMBER_TRIALS}
started_at=$(date --iso-8601=seconds)
EOF
}

stop_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping ${label} (pid ${pid})"
    kill "${pid}" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Force killing ${label} (pid ${pid})"
      kill -9 "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_pid "${RECORDER_PID}" "recorder"
  stop_pid "${POLICY_PID}" "policy"
  stop_pid "${ENGINE_PID}" "engine"
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

start_background() {
  local name="$1"
  local logfile="$2"
  shift 2
  echo "Starting ${name}" >&2
  "$@" >"${logfile}" 2>&1 &
  local pid=$!
  echo "  pid: ${pid}" >&2
  echo "  log: ${logfile}" >&2
  echo "${pid}" > "${LOG_ROOT}/${name}.pid"
  echo "${pid}"
}

check_alive() {
  local pid="$1"
  local label="$2"
  local logfile="$3"
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "${label} exited early. See ${logfile}" >&2
    tail -n 80 "${logfile}" >&2 || true
    exit 1
  fi
}

wait_for_recorder_completion() {
  local deadline=$((SECONDS + RESULT_TIMEOUT_SECONDS))
  local engine_exit_seen=0
  local engine_exit_started_at=0
  while (( SECONDS < deadline )); do
    if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
      wait "${RECORDER_PID}" 2>/dev/null || true
      return 0
    fi

    if ! kill -0 "${ENGINE_PID}" 2>/dev/null; then
      if (( engine_exit_seen == 0 )); then
        engine_exit_seen=1
        engine_exit_started_at="${SECONDS}"
        echo "Engine exited while recorder was still flushing; allowing ${AFTER_ENGINE_EXIT_GRACE_SECONDS}s grace for recorder finalization."
      elif (( SECONDS - engine_exit_started_at >= AFTER_ENGINE_EXIT_GRACE_SECONDS )); then
        echo "Recorder did not finish within ${AFTER_ENGINE_EXIT_GRACE_SECONDS}s after engine exit" >&2
        tail -n 120 "${RECORDER_LOG}" >&2 || true
        return 1
      fi
    fi

    sleep 1
  done

  echo "Recorder did not finish within ${RESULT_TIMEOUT_SECONDS}s" >&2
  tail -n 120 "${RECORDER_LOG}" >&2 || true
  return 1
}

wait_for_scoring_update() {
  local original_mtime="$1"
  local deadline=$((SECONDS + SCORING_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if [[ -f "${SCORING_OUTPUT_PATH}" ]]; then
      local current_mtime
      current_mtime="$(stat -c %Y "${SCORING_OUTPUT_PATH}" 2>/dev/null || echo 0)"
      if (( current_mtime > original_mtime )); then
        return 0
      fi
    fi
    sleep 1
  done
  return 1
}

postprocess_scoring_if_available() {
  local scoring_start_mtime="$1"
  local dataset_root="${DATA_ROOT}/outputs/gazebo_data/${RECORD_SESSION_NAME}/parquet_root/experiment_1"
  local dataset_scoring_yaml="${dataset_root}/scoring.yaml"
  local session_scoring_yaml="${SESSION_ROOT}/scoring.yaml"
  local have_scoring=0

  if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset root not found; skipping scoring copy: ${dataset_root}" >&2
    return 0
  fi

  mkdir -p "${dataset_root}"

  if wait_for_scoring_update "${scoring_start_mtime}"; then
    cp "${SCORING_OUTPUT_PATH}" "${dataset_scoring_yaml}"
    cp "${SCORING_OUTPUT_PATH}" "${session_scoring_yaml}"
    have_scoring=1
    echo "Copied scoring.yaml to ${dataset_scoring_yaml}"
  elif [[ -f "${SCORING_EXTRACTOR}" ]]; then
    if python3 "${SCORING_EXTRACTOR}" "${ENGINE_LOG}" "${dataset_scoring_yaml}" >/dev/null 2>&1; then
      cp "${dataset_scoring_yaml}" "${session_scoring_yaml}"
      have_scoring=1
      echo "Extracted scoring.yaml from engine.log to ${dataset_scoring_yaml}"
    fi
  fi

  if (( have_scoring == 0 )); then
    echo "No scoring data available after case run; skipping score postprocess." >&2
    return 0
  fi

  bash "${DATA_ROOT}/bash/gazebo_postprocess_scoring.sh" --dataset-root "${dataset_root}"
}

copy_teacher_stage_trace_if_available() {
  local dataset_root="${DATA_ROOT}/outputs/gazebo_data/${RECORD_SESSION_NAME}/parquet_root/experiment_1"
  local dataset_stage_trace_path="${dataset_root}/teacher_stage_trace.json"

  if [[ ! -f "${STAGE_TRACE_PATH}" ]]; then
    echo "Teacher stage trace not found; skipping copy: ${STAGE_TRACE_PATH}" >&2
    return 0
  fi

  if [[ ! -d "${dataset_root}" ]]; then
    echo "Dataset root not found; skipping teacher stage trace copy: ${dataset_root}" >&2
    return 0
  fi

  cp "${STAGE_TRACE_PATH}" "${dataset_stage_trace_path}"
  echo "Copied teacher stage trace to ${dataset_stage_trace_path}"
}

cd "${REPO_ROOT}"

RECORD_SESSION_NAME="${RECORD_SESSION_NAME:-${SESSION_NAME}}"
write_manifest

scoring_start_mtime=0
if [[ -f "${SCORING_OUTPUT_PATH}" ]]; then
  scoring_start_mtime="$(stat -c %Y "${SCORING_OUTPUT_PATH}" 2>/dev/null || echo 0)"
fi

if (( AUTO_CLEAN_STALE == 1 )); then
  bash "${RESET_SCRIPT}" >/dev/null 2>&1 || true
  sleep "${CLEAN_SETTLE_SECONDS}"
fi

ENGINE_PID="$(start_background engine "${ENGINE_LOG}" env \
  AIC_ENGINE_CONFIG_FILE="${CONFIG_FILE}" \
  SPAWN_TASK_BOARD=false \
  SPAWN_CABLE=false \
  GROUND_TRUTH=true \
  START_AIC_ENGINE=true \
  SHUTDOWN_ON_AIC_ENGINE_EXIT=false \
  SKIP_MODEL_READY=false \
  SKIP_READY_SIMULATOR=false \
  GAZEBO_GUI="${GAZEBO_GUI:-false}" \
  LAUNCH_RVIZ="${LAUNCH_RVIZ:-false}" \
  bash "${REPO_ROOT}/quantarune_teleop/bash/gazebo_engine_teleop.sh")"

sleep "${ENGINE_WAIT_SECONDS}"
check_alive "${ENGINE_PID}" "engine" "${ENGINE_LOG}"

RECORDER_PID="$(start_background recorder "${RECORDER_LOG}" \
  bash "${DATA_ROOT}/bash/gazebo_data_record.sh" \
  --session-name "${RECORD_SESSION_NAME}" \
  --number-trials "${RECORD_NUMBER_TRIALS}")"

sleep "${RECORDER_READY_SECONDS}"
check_alive "${RECORDER_PID}" "recorder" "${RECORDER_LOG}"

case "${POLICY_MODE}" in
  full)
    POLICY_PID="$(start_background policy "${POLICY_LOG}" \
      env AIC_CHEATCODE_STAGE_TRACE_PATH="${STAGE_TRACE_PATH}" \
      bash "${DATA_ROOT}/bash/gazebo_cheatcode_policy.sh" \
      --policy "${POLICY_IMPORT}")"
    ;;
  handoff|auto_rescue)
    POLICY_PID="$(start_background policy "${POLICY_LOG}" \
      env AIC_CHEATCODE_STAGE_TRACE_PATH="${STAGE_TRACE_PATH}" \
      AIC_CHEATCODE_MODE="${POLICY_MODE}" \
      bash "${DATA_ROOT}/bash/gazebo_intervention_policy.sh" \
      --policy "${POLICY_IMPORT}" \
      --handoff-z-offset-m "${HANDOFF_Z_OFFSET_M}" \
      --handoff-settle-s "${HANDOFF_SETTLE_S}")"
    ;;
  *)
    echo "Unknown --policy-mode: ${POLICY_MODE}" >&2
    echo "Expected one of: full, handoff, auto_rescue" >&2
    exit 2
    ;;
esac

sleep "${POLICY_WAIT_SECONDS}"
check_alive "${ENGINE_PID}" "engine" "${ENGINE_LOG}"
check_alive "${POLICY_PID}" "policy" "${POLICY_LOG}"

echo
echo "Official-engine CheatCode case is running."
echo "Session root: ${SESSION_ROOT}"
echo "Config: ${CONFIG_FILE}"
echo "Logs:"
echo "  engine:   ${ENGINE_LOG}"
echo "  recorder: ${RECORDER_LOG}"
echo "  policy:   ${POLICY_LOG}"
echo "  dataset:  ${DATA_ROOT}/outputs/gazebo_data/${RECORD_SESSION_NAME}"
echo "  scoring:  ${SCORING_OUTPUT_PATH}"
echo "Policy mode: ${POLICY_MODE}"
echo "Policy import: ${POLICY_IMPORT}"
echo

wait_for_recorder_completion
sleep "${RECORDER_FLUSH_SECONDS}"
postprocess_scoring_if_available "${scoring_start_mtime}"
copy_teacher_stage_trace_if_available
