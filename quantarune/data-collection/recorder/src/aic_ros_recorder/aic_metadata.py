import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aic_task_interfaces.msg import Task

ROBOT_TYPE = "ur5e_aic"

CAMERA_SHAPES: dict[str, tuple[int, int, int]] = {
    "left_camera": (1024, 1152, 3),
    "center_camera": (1024, 1152, 3),
    "right_camera": (1024, 1152, 3),
}

OBSERVATION_STATE_FEATURES: dict[str, type] = {
    "tcp_pose.position.x": float,
    "tcp_pose.position.y": float,
    "tcp_pose.position.z": float,
    "tcp_pose.orientation.x": float,
    "tcp_pose.orientation.y": float,
    "tcp_pose.orientation.z": float,
    "tcp_pose.orientation.w": float,
    "tcp_velocity.linear.x": float,
    "tcp_velocity.linear.y": float,
    "tcp_velocity.linear.z": float,
    "tcp_velocity.angular.x": float,
    "tcp_velocity.angular.y": float,
    "tcp_velocity.angular.z": float,
    "tcp_error.x": float,
    "tcp_error.y": float,
    "tcp_error.z": float,
    "tcp_error.rx": float,
    "tcp_error.ry": float,
    "tcp_error.rz": float,
    "joint_positions.0": float,
    "joint_positions.1": float,
    "joint_positions.2": float,
    "joint_positions.3": float,
    "joint_positions.4": float,
    "joint_positions.5": float,
    "joint_positions.6": float,
    "joint_velocity.0": float,
    "joint_velocity.1": float,
    "joint_velocity.2": float,
    "joint_velocity.3": float,
    "joint_velocity.4": float,
    "joint_velocity.5": float,
    "joint_velocity.6": float,
    "wrench.force.x": float,
    "wrench.force.y": float,
    "wrench.force.z": float,
    "wrench.torque.x": float,
    "wrench.torque.y": float,
    "wrench.torque.z": float,
    "task.target_valid": float,
    "task.cable_type_id": float,
    "task.plug_type_id": float,
    "task.port_type_id": float,
    "task.target_module_id": float,
    "task.target_port_id": float,
    "task.target_module_index": float,
    "task.target_port_index": float,
    "task.time_limit": float,
}

OBSERVATION_FEATURES: dict[str, type | tuple[int, int, int]] = {
    **OBSERVATION_STATE_FEATURES,
    **CAMERA_SHAPES,
}

TASK_TEXT_FEATURES: dict[str, type] = {
    "task.command": str,
    "task.id": str,
    "task.cable_type": str,
    "task.cable_name": str,
    "task.plug_type": str,
    "task.plug_name": str,
    "task.port_type": str,
    "task.port_name": str,
    "task.target_module_name": str,
}

CABLE_TYPE_TO_ID = {
    "unknown": 0,
    "sfp_sc": 1,
}

PLUG_TYPE_TO_ID = {
    "unknown": 0,
    "sc": 1,
    "sfp": 2,
    "lc": 3,
}

PORT_TYPE_TO_ID = {
    "unknown": 0,
    "sc": 1,
    "sfp": 2,
}

MODULE_NAME_TO_ID = {
    "unknown": 0,
    "nic_card_mount_0": 1,
    "nic_card_mount_1": 2,
    "nic_card_mount_2": 3,
    "nic_card_mount_3": 4,
    "nic_card_mount_4": 5,
    "sc_port_0": 6,
    "sc_port_1": 7,
}

PORT_NAME_TO_ID = {
    "unknown": 0,
    "sfp_port_0": 1,
    "sfp_port_1": 2,
    "sc_port_base": 3,
}


def task_msg_to_text(task_msg: "Task") -> str:
    return (
        f"insert {task_msg.cable_type} cable {task_msg.cable_name} "
        f"with {task_msg.plug_type} plug {task_msg.plug_name} "
        f"into {task_msg.port_type} port {task_msg.port_name} "
        f"on module {task_msg.target_module_name}"
    )


def encode_task_text(task_msg: "Task") -> dict[str, str]:
    return {
        "task.command": task_msg_to_text(task_msg),
        "task.id": task_msg.id,
        "task.cable_type": task_msg.cable_type,
        "task.cable_name": task_msg.cable_name,
        "task.plug_type": task_msg.plug_type,
        "task.plug_name": task_msg.plug_name,
        "task.port_type": task_msg.port_type,
        "task.port_name": task_msg.port_name,
        "task.target_module_name": task_msg.target_module_name,
    }


def task_text_dataset_features(prefix: str) -> dict[str, dict[str, Any]]:
    return {
        (f"{prefix}.{key}" if prefix else key): {
            "dtype": "string",
            "shape": (1,),
            "names": None,
        }
        for key in TASK_TEXT_FEATURES
    }


def extract_module_index(name: str) -> int:
    match = re.search(r"(\d+)$", name or "")
    return int(match.group(1)) if match else -1


def extract_port_index(name: str) -> int:
    match = re.search(r"(\d+)$", name or "")
    return int(match.group(1)) if match else -1


def encode_task_numeric(task_msg: "Task") -> dict[str, float]:
    cable_type_id = CABLE_TYPE_TO_ID.get(task_msg.cable_type, 0)
    plug_type_id = PLUG_TYPE_TO_ID.get(task_msg.plug_type, 0)
    port_type_id = PORT_TYPE_TO_ID.get(task_msg.port_type, 0)
    target_module_id = MODULE_NAME_TO_ID.get(task_msg.target_module_name, 0)
    target_port_id = PORT_NAME_TO_ID.get(task_msg.port_name, 0)
    module_index = extract_module_index(task_msg.target_module_name)
    port_index = extract_port_index(task_msg.port_name)

    return {
        "task.target_valid": 1.0,
        "task.cable_type_id": float(cable_type_id),
        "task.plug_type_id": float(plug_type_id),
        "task.port_type_id": float(port_type_id),
        "task.target_module_id": float(target_module_id),
        "task.target_port_id": float(target_port_id),
        "task.target_module_index": float(module_index),
        "task.target_port_index": float(port_index),
        "task.time_limit": float(task_msg.time_limit),
    }
