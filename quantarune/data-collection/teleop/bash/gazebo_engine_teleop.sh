#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export DBX_CONTAINER_MANAGER="${DBX_CONTAINER_MANAGER:-docker}"

SPAWN_TASK_BOARD_VALUE="${SPAWN_TASK_BOARD:-true}"
SPAWN_CABLE_VALUE="${SPAWN_CABLE:-true}"
START_AIC_ENGINE_VALUE="${START_AIC_ENGINE:-true}"
GAZEBO_GUI_VALUE="${GAZEBO_GUI:-true}"
LAUNCH_RVIZ_VALUE="${LAUNCH_RVIZ:-true}"

# Do not allow engine-managed trials and manual scene spawning at the same time.
# That combination can leave duplicate boards/cables in Gazebo because both layers
# try to own task lifecycle.
if [[ "${START_AIC_ENGINE_VALUE}" == "true" ]] && [[ "${SPAWN_TASK_BOARD_VALUE}" == "true" || "${SPAWN_CABLE_VALUE}" == "true" ]]; then
  echo "Invalid launch combination in gazebo_engine_teleop.sh" >&2
  echo "START_AIC_ENGINE=true means aic_engine owns task spawning." >&2
  echo "Set SPAWN_TASK_BOARD=false and SPAWN_CABLE=false, or set START_AIC_ENGINE=false." >&2
  exit 2
fi

AIC_EVAL_CONTAINER="${AIC_EVAL_CONTAINER:-aic_eval}"
AIC_EVAL_ENTER_ROOT="${AIC_EVAL_ENTER_ROOT:-false}"
AIC_ENGINE_CONFIG_FILE_VALUE="${AIC_ENGINE_CONFIG_FILE:-}"
LOCAL_CONTROLLER_CONFIG="${REPO_ROOT}/aic_bringup/config/aic_ros2_controllers.yaml"
CONTAINER_CONTROLLER_CONFIG="/ws_aic/install/share/aic_bringup/config/aic_ros2_controllers.yaml"

if ! distrobox list 2>/dev/null | awk -F'|' -v name="${AIC_EVAL_CONTAINER}" '
  NR > 1 {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
    if ($2 == name) {
      found = 1
    }
  }
  END {
    exit found ? 0 : 1
  }
'; then
  cat >&2 <<EOF
Missing distrobox container '${AIC_EVAL_CONTAINER}'.

Create it once with:
  export DBX_CONTAINER_MANAGER=docker
  docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
  distrobox create -r --nvidia --image ghcr.io/intrinsic-dev/aic/aic_eval:latest --name ${AIC_EVAL_CONTAINER}

After that, rerun the same smoke or teleop command.
EOF
  exit 3
fi

# Keep the running evaluation container aligned with the repo-side controller tuning
# so teleop tests use the latest local impedance settings without rebuilding the image.
if docker ps -a --format "{{.Names}}" | grep -Fxq "${AIC_EVAL_CONTAINER}"; then
  docker cp "${LOCAL_CONTROLLER_CONFIG}" "${AIC_EVAL_CONTAINER}:${CONTAINER_CONTROLLER_CONFIG}"
fi

ENTRYPOINT_ARGS=(
  "spawn_task_board:=${SPAWN_TASK_BOARD_VALUE}"
  "task_board_x:=${TASK_BOARD_X:-0.3}"
  "task_board_y:=${TASK_BOARD_Y:--0.1}"
  "task_board_z:=${TASK_BOARD_Z:-1.2}"
  "task_board_roll:=${TASK_BOARD_ROLL:-0.0}"
  "task_board_pitch:=${TASK_BOARD_PITCH:-0.0}"
  "task_board_yaw:=${TASK_BOARD_YAW:-0.785}"
  "sfp_mount_rail_0_present:=${SFP_MOUNT_RAIL_0_PRESENT:-true}"
  "sfp_mount_rail_0_translation:=${SFP_MOUNT_RAIL_0_TRANSLATION:--0.08}"
  "sfp_mount_rail_0_roll:=${SFP_MOUNT_RAIL_0_ROLL:-0.0}"
  "sfp_mount_rail_0_pitch:=${SFP_MOUNT_RAIL_0_PITCH:-0.0}"
  "sfp_mount_rail_0_yaw:=${SFP_MOUNT_RAIL_0_YAW:-0.0}"
  "sc_mount_rail_0_present:=${SC_MOUNT_RAIL_0_PRESENT:-true}"
  "sc_mount_rail_0_translation:=${SC_MOUNT_RAIL_0_TRANSLATION:--0.09}"
  "sc_mount_rail_0_roll:=${SC_MOUNT_RAIL_0_ROLL:-0.0}"
  "sc_mount_rail_0_pitch:=${SC_MOUNT_RAIL_0_PITCH:-0.0}"
  "sc_mount_rail_0_yaw:=${SC_MOUNT_RAIL_0_YAW:-0.0}"
  "sfp_mount_rail_1_present:=${SFP_MOUNT_RAIL_1_PRESENT:-false}"
  "sfp_mount_rail_1_translation:=${SFP_MOUNT_RAIL_1_TRANSLATION:-0.0}"
  "sfp_mount_rail_1_roll:=${SFP_MOUNT_RAIL_1_ROLL:-0.0}"
  "sfp_mount_rail_1_pitch:=${SFP_MOUNT_RAIL_1_PITCH:-0.0}"
  "sfp_mount_rail_1_yaw:=${SFP_MOUNT_RAIL_1_YAW:-0.0}"
  "sc_mount_rail_1_present:=${SC_MOUNT_RAIL_1_PRESENT:-false}"
  "sc_mount_rail_1_translation:=${SC_MOUNT_RAIL_1_TRANSLATION:-0.0}"
  "sc_mount_rail_1_roll:=${SC_MOUNT_RAIL_1_ROLL:-0.0}"
  "sc_mount_rail_1_pitch:=${SC_MOUNT_RAIL_1_PITCH:-0.0}"
  "sc_mount_rail_1_yaw:=${SC_MOUNT_RAIL_1_YAW:-0.0}"
  "nic_card_mount_0_present:=${NIC_CARD_MOUNT_0_PRESENT:-true}"
  "nic_card_mount_0_translation:=${NIC_CARD_MOUNT_0_TRANSLATION:-0.005}"
  "nic_card_mount_0_roll:=${NIC_CARD_MOUNT_0_ROLL:-0.0}"
  "nic_card_mount_0_pitch:=${NIC_CARD_MOUNT_0_PITCH:-0.0}"
  "nic_card_mount_0_yaw:=${NIC_CARD_MOUNT_0_YAW:-0.0}"
  "nic_card_mount_1_present:=${NIC_CARD_MOUNT_1_PRESENT:-false}"
  "nic_card_mount_1_translation:=${NIC_CARD_MOUNT_1_TRANSLATION:-0.0}"
  "nic_card_mount_1_roll:=${NIC_CARD_MOUNT_1_ROLL:-0.0}"
  "nic_card_mount_1_pitch:=${NIC_CARD_MOUNT_1_PITCH:-0.0}"
  "nic_card_mount_1_yaw:=${NIC_CARD_MOUNT_1_YAW:-0.0}"
  "nic_card_mount_2_present:=${NIC_CARD_MOUNT_2_PRESENT:-false}"
  "nic_card_mount_2_translation:=${NIC_CARD_MOUNT_2_TRANSLATION:-0.0}"
  "nic_card_mount_2_roll:=${NIC_CARD_MOUNT_2_ROLL:-0.0}"
  "nic_card_mount_2_pitch:=${NIC_CARD_MOUNT_2_PITCH:-0.0}"
  "nic_card_mount_2_yaw:=${NIC_CARD_MOUNT_2_YAW:-0.0}"
  "nic_card_mount_3_present:=${NIC_CARD_MOUNT_3_PRESENT:-false}"
  "nic_card_mount_3_translation:=${NIC_CARD_MOUNT_3_TRANSLATION:-0.0}"
  "nic_card_mount_3_roll:=${NIC_CARD_MOUNT_3_ROLL:-0.0}"
  "nic_card_mount_3_pitch:=${NIC_CARD_MOUNT_3_PITCH:-0.0}"
  "nic_card_mount_3_yaw:=${NIC_CARD_MOUNT_3_YAW:-0.0}"
  "nic_card_mount_4_present:=${NIC_CARD_MOUNT_4_PRESENT:-false}"
  "nic_card_mount_4_translation:=${NIC_CARD_MOUNT_4_TRANSLATION:-0.0}"
  "nic_card_mount_4_roll:=${NIC_CARD_MOUNT_4_ROLL:-0.0}"
  "nic_card_mount_4_pitch:=${NIC_CARD_MOUNT_4_PITCH:-0.0}"
  "nic_card_mount_4_yaw:=${NIC_CARD_MOUNT_4_YAW:-0.0}"
  "sc_port_0_present:=${SC_PORT_0_PRESENT:-true}"
  "sc_port_0_translation:=${SC_PORT_0_TRANSLATION:--0.04}"
  "sc_port_0_roll:=${SC_PORT_0_ROLL:-0.0}"
  "sc_port_0_pitch:=${SC_PORT_0_PITCH:-0.0}"
  "sc_port_0_yaw:=${SC_PORT_0_YAW:-0.0}"
  "sc_port_1_present:=${SC_PORT_1_PRESENT:-false}"
  "sc_port_1_translation:=${SC_PORT_1_TRANSLATION:-0.0}"
  "sc_port_1_roll:=${SC_PORT_1_ROLL:-0.0}"
  "sc_port_1_pitch:=${SC_PORT_1_PITCH:-0.0}"
  "sc_port_1_yaw:=${SC_PORT_1_YAW:-0.0}"
  "spawn_cable:=${SPAWN_CABLE_VALUE}"
  "cable_type:=${CABLE_TYPE:-sfp_sc_cable}"
  "attach_cable_to_gripper:=${ATTACH_CABLE_TO_GRIPPER:-true}"
  "ground_truth:=${GROUND_TRUTH:-true}"
  "start_aic_engine:=${START_AIC_ENGINE_VALUE}"
  "shutdown_on_aic_engine_exit:=${SHUTDOWN_ON_AIC_ENGINE_EXIT:-false}"
  "skip_model_ready:=${SKIP_MODEL_READY:-true}"
  "skip_ready_simulator:=${SKIP_READY_SIMULATOR:-true}"
  "gazebo_gui:=${GAZEBO_GUI_VALUE}"
  "launch_rviz:=${LAUNCH_RVIZ_VALUE}"
)

if [[ -n "${AIC_ENGINE_CONFIG_FILE_VALUE}" ]]; then
  ENTRYPOINT_ARGS+=("aic_engine_config_file:=${AIC_ENGINE_CONFIG_FILE_VALUE}")
fi

DISTROBOX_ENTER_CMD=(distrobox enter "${AIC_EVAL_CONTAINER}" --)
if [[ "${AIC_EVAL_ENTER_ROOT}" == "true" ]]; then
  DISTROBOX_ENTER_CMD=(distrobox enter -r "${AIC_EVAL_CONTAINER}" --)
fi

"${DISTROBOX_ENTER_CMD[@]}" /entrypoint.sh "${ENTRYPOINT_ARGS[@]}"
