#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/quantarune_data"
AIC_EVAL_CONTAINER="${AIC_EVAL_CONTAINER:-aic_eval}"

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

stop_matching_processes() {
  local pattern="$1"
  local label="$2"
  local pids=""
  pids="$(pgrep -f "${pattern}" || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    stop_pid "${pid}" "${label}"
  done <<< "${pids}"
}

cd "${REPO_ROOT}"

echo "Stopping host-side fixed-scene policy/task processes..."
bash "${DATA_ROOT}/bash/stop_fixed_scene_cheatcode.sh" >/dev/null 2>&1 || true

echo "Stopping host-side teleop/recorder wrappers..."
stop_matching_processes "quantarune_teleop/scripts/gazebo_teleop.py" "gazebo teleop"
stop_matching_processes "joint_keyboard_teleop.py" "joint teleop"
stop_matching_processes "cartesian_keyboard_teleop.py" "cartesian teleop"
stop_matching_processes "aic_ros_recorder_node" "AIC ROS recorder"
stop_matching_processes "quantarune_data/scripts/gazebo_data.py record" "data recorder wrapper"
stop_matching_processes "quantarune_data/scripts/gazebo_data.py intervention-policy" "intervention policy wrapper"
stop_matching_processes "quantarune_data/scripts/gazebo_data.py cheatcode" "cheatcode wrapper"
stop_matching_processes "distrobox enter -r ${AIC_EVAL_CONTAINER} -- /entrypoint.sh" "eval entrypoint wrapper"
stop_matching_processes "distrobox enter ${AIC_EVAL_CONTAINER} -- /entrypoint.sh" "eval entrypoint wrapper"
stop_matching_processes "/entrypoint.sh" "eval entrypoint"
stop_matching_processes "ros2 run aic_model aic_model" "aic_model"
stop_matching_processes " aic_model " "aic_model"
stop_matching_processes "rviz2" "RViz"
stop_matching_processes "gz sim" "Gazebo sim"
stop_matching_processes "gz_server" "Gazebo server"
stop_matching_processes "ros_gz_container" "ROS-Gazebo container"
stop_matching_processes "component_container" "ROS component container"

if docker ps --format "{{.Names}}" | grep -Fxq "${AIC_EVAL_CONTAINER}"; then
  echo "Stopping Docker container ${AIC_EVAL_CONTAINER}"
  docker stop "${AIC_EVAL_CONTAINER}" >/dev/null
else
  echo "Docker container ${AIC_EVAL_CONTAINER} is already stopped"
fi

for _ in {1..20}; do
  if ! docker ps --format "{{.Names}}" | grep -Fxq "${AIC_EVAL_CONTAINER}"; then
    break
  fi
  sleep 0.5
done

sleep "${AIC_EVAL_RESET_SETTLE_SECONDS:-8}"

echo
echo "Gazebo eval stack reset complete."
echo "Next launch should start from a clean container and avoid duplicate arms / duplicate RViz."
