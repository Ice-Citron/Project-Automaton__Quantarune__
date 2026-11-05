from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

from aic_example_policies.ros.RunACT import RunACT, _encode_task_numeric
from aic_example_policies.ros.vision_target_features import (
    CenterPortFeatureExtractor,
    VISION_FEATURE_NAMES,
    build_task_grounding_metadata,
)


DEFAULT_48D_STATE_FEATURE_NAMES = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
    "joint_velocity.0",
    "joint_velocity.1",
    "joint_velocity.2",
    "joint_velocity.3",
    "joint_velocity.4",
    "joint_velocity.5",
    "joint_velocity.6",
    "wrench.force.x",
    "wrench.force.y",
    "wrench.force.z",
    "wrench.torque.x",
    "wrench.torque.y",
    "wrench.torque.z",
    "task.target_valid",
    "task.cable_type_id",
    "task.plug_type_id",
    "task.port_type_id",
    "task.target_module_id",
    "task.target_port_id",
    "task.target_module_index",
    "task.target_port_index",
    "task.time_limit",
]
DEFAULT_56D_STATE_FEATURE_NAMES = [
    *DEFAULT_48D_STATE_FEATURE_NAMES,
    *VISION_FEATURE_NAMES,
]


class RunACTVision(RunACT):
    """ACT runtime wrapper that appends compact vision features to the state."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._insertion_event_seen = False
        self._vision_extractor = CenterPortFeatureExtractor()
        self._latest_task_grounding_metadata: dict[str, object] | None = None
        self.state_feature_names = list(DEFAULT_56D_STATE_FEATURE_NAMES)
        expected_state_dim = len(self.state_feature_names)
        if self.state_dim != expected_state_dim:
            raise RuntimeError(
                "RunACTVision expects a checkpoint trained with appended vision features. "
                f"Expected observation.state dim {expected_state_dim}, got {self.state_dim}. "
                "This usually means a 48D ACT checkpoint was routed through the ACT56 runtime."
            )
        self._validate_checkpoint_feature_contract()
        self.get_logger().info(
            "RunACTVision enabled with appended center-camera vision features "
            f"(state_dim={self.state_dim})."
        )

    def insert_cable(self, *args, **kwargs):
        self._insertion_event_seen = False
        return super().insert_cable(*args, **kwargs)

    def get_latest_task_grounding_metadata(self) -> dict[str, object] | None:
        return self._latest_task_grounding_metadata

    def _validate_checkpoint_feature_contract(self) -> None:
        config = getattr(self.policy, "config", None)
        if config is None:
            return

        input_features = getattr(config, "input_features", None)
        if not isinstance(input_features, dict):
            return

        state_meta = input_features.get("observation.state")
        if state_meta is None:
            return

        if isinstance(state_meta, dict):
            names = state_meta.get("names")
            shape = state_meta.get("shape")
        else:
            names = getattr(state_meta, "names", None)
            shape = getattr(state_meta, "shape", None)

        if shape:
            expected_shape = (len(self.state_feature_names),)
            actual_shape = tuple(shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    "RunACTVision checkpoint metadata does not match ACT56 state shape. "
                    f"Expected {expected_shape}, got {actual_shape}."
                )

        if isinstance(names, list) and names and list(names) != self.state_feature_names:
            raise RuntimeError(
                "RunACTVision checkpoint metadata does not match the ACT56 feature contract. "
                f"Expected {self.state_feature_names}, got {names}."
            )

    def prepare_observations(
        self, obs_msg: Observation, task_msg: Task | None
    ) -> Dict[str, torch.Tensor]:
        obs = {
            "observation.images.left_camera": self._img_to_tensor(
                obs_msg.left_image,
                self.device,
                self.image_scaling,
                self.img_stats["left"]["mean"],
                self.img_stats["left"]["std"],
            ),
            "observation.images.center_camera": self._img_to_tensor(
                obs_msg.center_image,
                self.device,
                self.image_scaling,
                self.img_stats["center"]["mean"],
                self.img_stats["center"]["std"],
            ),
            "observation.images.right_camera": self._img_to_tensor(
                obs_msg.right_image,
                self.device,
                self.image_scaling,
                self.img_stats["right"]["mean"],
                self.img_stats["right"]["std"],
            ),
        }

        controller_state = obs_msg.controller_state
        tcp_pose = controller_state.tcp_pose
        tcp_vel = controller_state.tcp_velocity
        tare_wrench = controller_state.fts_tare_offset.wrench
        joint_positions = list(obs_msg.joint_states.position)
        joint_velocities = (
            list(obs_msg.joint_states.velocity)
            if obs_msg.joint_states.velocity
            else []
        )
        task_features = _encode_task_numeric(task_msg)
        self._latest_task_grounding_metadata = build_task_grounding_metadata(
            task_msg=task_msg
        )
        vision_features = self._vision_extractor.extract(
            raw_img=obs_msg.center_image,
            task_msg=task_msg,
        ).as_dict()

        state_dict = {
            "tcp_pose.position.x": tcp_pose.position.x,
            "tcp_pose.position.y": tcp_pose.position.y,
            "tcp_pose.position.z": tcp_pose.position.z,
            "tcp_pose.orientation.x": tcp_pose.orientation.x,
            "tcp_pose.orientation.y": tcp_pose.orientation.y,
            "tcp_pose.orientation.z": tcp_pose.orientation.z,
            "tcp_pose.orientation.w": tcp_pose.orientation.w,
            "tcp_velocity.linear.x": tcp_vel.linear.x,
            "tcp_velocity.linear.y": tcp_vel.linear.y,
            "tcp_velocity.linear.z": tcp_vel.linear.z,
            "tcp_velocity.angular.x": tcp_vel.angular.x,
            "tcp_velocity.angular.y": tcp_vel.angular.y,
            "tcp_velocity.angular.z": tcp_vel.angular.z,
            "tcp_error.x": controller_state.tcp_error[0],
            "tcp_error.y": controller_state.tcp_error[1],
            "tcp_error.z": controller_state.tcp_error[2],
            "tcp_error.rx": controller_state.tcp_error[3],
            "tcp_error.ry": controller_state.tcp_error[4],
            "tcp_error.rz": controller_state.tcp_error[5],
            "joint_positions.0": joint_positions[0] if len(joint_positions) > 0 else 0.0,
            "joint_positions.1": joint_positions[1] if len(joint_positions) > 1 else 0.0,
            "joint_positions.2": joint_positions[2] if len(joint_positions) > 2 else 0.0,
            "joint_positions.3": joint_positions[3] if len(joint_positions) > 3 else 0.0,
            "joint_positions.4": joint_positions[4] if len(joint_positions) > 4 else 0.0,
            "joint_positions.5": joint_positions[5] if len(joint_positions) > 5 else 0.0,
            "joint_positions.6": joint_positions[6] if len(joint_positions) > 6 else 0.0,
            "joint_velocity.0": joint_velocities[0] if len(joint_velocities) > 0 else 0.0,
            "joint_velocity.1": joint_velocities[1] if len(joint_velocities) > 1 else 0.0,
            "joint_velocity.2": joint_velocities[2] if len(joint_velocities) > 2 else 0.0,
            "joint_velocity.3": joint_velocities[3] if len(joint_velocities) > 3 else 0.0,
            "joint_velocity.4": joint_velocities[4] if len(joint_velocities) > 4 else 0.0,
            "joint_velocity.5": joint_velocities[5] if len(joint_velocities) > 5 else 0.0,
            "joint_velocity.6": joint_velocities[6] if len(joint_velocities) > 6 else 0.0,
            "wrench.force.x": tare_wrench.force.x,
            "wrench.force.y": tare_wrench.force.y,
            "wrench.force.z": tare_wrench.force.z,
            "wrench.torque.x": tare_wrench.torque.x,
            "wrench.torque.y": tare_wrench.torque.y,
            "wrench.torque.z": tare_wrench.torque.z,
            **task_features,
            **vision_features,
        }

        state_np = np.array(
            [state_dict[name] for name in self.state_feature_names],
            dtype=np.float32,
        )
        raw_state_tensor = (
            torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        )
        if not np.isfinite(state_np).all():
            raw_state_tensor = torch.from_numpy(
                np.nan_to_num(state_np, nan=0.0, posinf=0.0, neginf=0.0)
            ).float().unsqueeze(0).to(self.device)

        obs["observation.state"] = (raw_state_tensor - self.state_mean) / self.state_std
        if not torch.isfinite(obs["observation.state"]).all():
            obs["observation.state"] = torch.nan_to_num(obs["observation.state"])

        return obs
