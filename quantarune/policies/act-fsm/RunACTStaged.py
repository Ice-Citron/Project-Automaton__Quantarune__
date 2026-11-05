#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import annotations

import os
import time
from collections import deque
from typing import Sequence

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from rclpy.time import Time
from tf2_ros import TransformException

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .RunACTCompetitor import (
    RunACTCompetitor,
    _parse_bool,
    _parse_float,
    _parse_float_list,
)
from .sfp_tuning_profiles import (
    get_sfp_primitive_tuning_profile,
    is_sfp_gentle_reference_tuning,
)


def _parse_float_alias(primary: str, alias: str, default: float) -> float:
    value = os.environ.get(primary)
    if value is None:
        value = os.environ.get(alias)
    return float(value) if value is not None else default


def _parse_bool_alias(primary: str, alias: str, default: bool) -> bool:
    value = os.environ.get(primary)
    if value is None:
        value = os.environ.get(alias)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_list_alias(
    primary: str, alias: str, default: Sequence[float]
) -> list[float]:
    value = os.environ.get(primary)
    if value is None:
        value = os.environ.get(alias)
    if value is None:
        return list(default)
    parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parsed) != len(default):
        raise ValueError(
            f"{primary}/{alias} expected {len(default)} comma-separated floats, got {len(parsed)}"
        )
    return parsed


def _quat_xyzw_to_array(quaternion: Quaternion) -> np.ndarray:
    quat = np.array(
        [
            float(quaternion.x),
            float(quaternion.y),
            float(quaternion.z),
            float(quaternion.w),
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_xyzw_inverse(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64)


def _quat_xyzw_to_msg(quat: np.ndarray) -> Quaternion:
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-9)
    return Quaternion(
        x=float(quat[0]),
        y=float(quat[1]),
        z=float(quat[2]),
        w=float(quat[3]),
    )


class RunACTStaged(RunACTCompetitor):
    """Pose-preserving ACT runtime with staged approach / align / guarded insert / retry."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.sfp_primitive_tuning_profile = get_sfp_primitive_tuning_profile()
        self._sfp_gentle_reference_tuning = is_sfp_gentle_reference_tuning(
            self.sfp_primitive_tuning_profile
        )
        self.stage_policy_routes = {
            "APPROACH": self._resolve_stage_policy_route(
                phase="APPROACH",
                env_var="AIC_ACT_STAGE_ROUTE_APPROACH",
                default_route="approach",
            ),
            "ALIGN": self._resolve_stage_policy_route(
                phase="ALIGN",
                env_var="AIC_ACT_STAGE_ROUTE_ALIGN",
                default_route="align",
            ),
            "GUARDED_INSERT": self._resolve_stage_policy_route(
                phase="GUARDED_INSERT",
                env_var="AIC_ACT_STAGE_ROUTE_GUARDED_INSERT",
                default_route="guarded_insert",
            ),
            "RETRY_OR_FINISH": self._resolve_stage_policy_route(
                phase="RETRY_OR_FINISH",
                env_var="AIC_ACT_STAGE_ROUTE_RETRY_OR_FINISH",
                default_route="retry_or_finish",
            ),
        }
        self.enable_staged_wrapper = _parse_bool_alias(
            "AIC_ACT_STAGED_ENABLE_STAGED_WRAPPER",
            "enable_staged_wrapper",
            True,
        )
        self.approach_offset = np.asarray(
            _parse_float_list_alias(
                "AIC_ACT_STAGED_APPROACH_OFFSET",
                "approach_offset",
                [0.0, 0.0, 0.0],
            ),
            dtype=np.float64,
        )
        self.insertion_speed_scale = _parse_float_alias(
            "AIC_ACT_STAGED_INSERTION_SPEED_SCALE",
            "insertion_speed_scale",
            0.35,
        )
        self.guarded_insert_force_start_n = _parse_float_alias(
            "AIC_ACT_STAGED_GUARDED_INSERT_FORCE_START_N",
            "guarded_insert_force_start",
            self.force_rampdown_start_n,
        )
        self.guarded_insert_force_stop_n = _parse_float_alias(
            "AIC_ACT_STAGED_GUARDED_INSERT_FORCE_STOP_N",
            "guarded_insert_force_stop",
            self.force_hard_stop_n,
        )
        self.max_retries = int(
            _parse_float_alias(
                "AIC_ACT_STAGED_RETRY_COUNT",
                "retry_count",
                1.0,
            )
        )
        self.backout_distance_m = _parse_float_alias(
            "AIC_ACT_STAGED_BACKOUT_DISTANCE_M",
            "backout_distance",
            0.010,
        )
        self.settle_time_s = _parse_float_alias(
            "AIC_ACT_STAGED_SETTLE_TIME_S",
            "settle_time",
            0.75,
        )
        self.require_insertion_event = _parse_bool_alias(
            "AIC_ACT_STAGED_REQUIRE_INSERTION_EVENT",
            "require_insertion_event",
            self.require_insertion_event,
        )
        self.approach_min_sec = _parse_float(
            "AIC_ACT_STAGED_APPROACH_MIN_SEC", 0.75
        )
        self.max_guarded_insert_sec = _parse_float(
            "AIC_ACT_STAGED_MAX_GUARDED_INSERT_SEC",
            6.00 if self._sfp_gentle_reference_tuning else 1.50,
        )
        self.align_position_tolerance_m = _parse_float(
            "AIC_ACT_STAGED_ALIGN_POSITION_TOL_M", 0.025
        )
        self.align_xy_error_tolerance_m = _parse_float(
            "AIC_ACT_STAGED_ALIGN_XY_ERROR_TOL_M", 0.015
        )
        self.align_angular_tolerance_rad = _parse_float(
            "AIC_ACT_STAGED_ALIGN_ANGULAR_TOL_RAD", 0.12
        )
        self.guard_progress_epsilon_m = _parse_float(
            "AIC_ACT_STAGED_PROGRESS_EPS_M",
            0.0004 if self._sfp_gentle_reference_tuning else 0.0015,
        )
        self.guard_position_smoothing_scale = _parse_float(
            "AIC_ACT_STAGED_ALIGN_SMOOTHING_SCALE", 0.25
        )
        self.geometry_debug_enabled = _parse_bool(
            "AIC_ACT_STAGED_GEOMETRY_DEBUG", False
        )
        self.privileged_terminal_target_enabled = _parse_bool(
            "AIC_ACT_STAGED_PRIVILEGED_TERMINAL_TARGET", False
        )
        self.privileged_hover_offset_m = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_HOVER_OFFSET_M", 0.160
        )
        self.privileged_align_offset_m = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_ALIGN_OFFSET_M", 0.130
        )
        self.privileged_insert_target_offset_m = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_INSERT_TARGET_OFFSET_M", 0.105
        )
        self.privileged_xy_integrator_gain = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_XY_INTEGRATOR_GAIN", 0.15
        )
        self.privileged_xy_direct_gain = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_XY_DIRECT_GAIN", 0.0
        )
        self.privileged_xy_integrator_windup_m = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_XY_INTEGRATOR_WINDUP_M", 0.050
        )
        self.privileged_ready_xy_tol_m = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_READY_XY_TOL_M", 0.006
        )
        self.privileged_guard_lateral_gain = _parse_float(
            "AIC_ACT_STAGED_PRIVILEGED_GUARD_LATERAL_GAIN", 0.60
        )
        self.privileged_debug_every_n = int(
            _parse_float("AIC_ACT_STAGED_PRIVILEGED_DEBUG_EVERY_N", 10.0)
        )
        self._last_guidance_context: dict[str, object] | None = None
        self._privileged_debug_counter = 0
        self._privileged_tip_xy_error_integrator = np.zeros(2, dtype=np.float64)
        self._recent_act_pose_positions: deque[np.ndarray] = deque(maxlen=6)
        self._phase = "APPROACH"
        self._phase_start_time = 0.0
        self._align_stable_since: float | None = None
        self._guard_direction: np.ndarray | None = None
        self._guard_start_pos: np.ndarray | None = None
        self._best_guard_progress_m = 0.0
        self._retry_target_pose: Pose | None = None
        self._retry_reason = ""
        self._retries_used = 0
        self._max_force_seen_n = 0.0
        self.get_logger().info(
            "RunACTStaged active with "
            f"enable_staged_wrapper={self.enable_staged_wrapper} "
            f"approach_offset={self.approach_offset.tolist()} "
            f"insertion_speed_scale={self.insertion_speed_scale:.3f} "
            f"guarded_force=({self.guarded_insert_force_start_n:.1f}, {self.guarded_insert_force_stop_n:.1f}) "
            f"retry_count={self.max_retries} "
            f"backout_distance_m={self.backout_distance_m:.3f} "
            f"settle_time_s={self.settle_time_s:.3f} "
            f"require_insertion_event={self.require_insertion_event} "
            f"sfp_tuning_profile={self.sfp_primitive_tuning_profile}"
        )
        self.get_logger().info(
            "RunACTStaged stage policy routes: "
            + ", ".join(
                f"{phase}={route or '<task-default>'}"
                for phase, route in self.stage_policy_routes.items()
            )
        )
        if self.privileged_terminal_target_enabled:
            self.get_logger().warning(
                "RunACTStaged privileged terminal target is ENABLED. "
                "This is for local geometry diagnosis only and is not submission-safe."
            )
        else:
            self.get_logger().info(
                "RunACTStaged uses ACT-led pose targets with staged heuristics. "
                "No explicit target geometry is available in the current Observation/Task path."
            )

    @staticmethod
    def _norm(values: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))

    @staticmethod
    def _pose_position_array(pose: Pose) -> np.ndarray:
        return np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
        )

    @staticmethod
    def _copy_pose(pose: Pose) -> Pose:
        return Pose(
            position=Point(
                x=float(pose.position.x),
                y=float(pose.position.y),
                z=float(pose.position.z),
            ),
            orientation=Quaternion(
                x=float(pose.orientation.x),
                y=float(pose.orientation.y),
                z=float(pose.orientation.z),
                w=float(pose.orientation.w),
            ),
        )

    @staticmethod
    def _normalized(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return fallback.copy()
        return vector / norm

    def _build_guidance_context(self, observation_msg, task: Task):
        if not (self.geometry_debug_enabled or self.privileged_terminal_target_enabled):
            self._last_guidance_context = None
            return None
        context = self._build_privileged_terminal_context(observation_msg, task)
        self._last_guidance_context = context
        return context

    def _apply_guidance_pose(
        self,
        pose: Pose,
        observation_msg,
        task: Task,
        guidance_context,
    ) -> Pose:
        if (
            not self.privileged_terminal_target_enabled
            or guidance_context is None
            or not guidance_context.get("available")
        ):
            return pose
        phase = str(getattr(self, "_phase", "APPROACH"))
        if phase == "APPROACH":
            target_pose = guidance_context.get("hover_pose")
        elif phase == "ALIGN":
            target_pose = guidance_context.get("align_pose")
        else:
            target_pose = guidance_context.get("insert_pose")
        return self._copy_pose(target_pose) if isinstance(target_pose, Pose) else pose

    def _guidance_allows_align(self, guidance_context) -> bool:
        if not self.privileged_terminal_target_enabled:
            return True
        return bool(
            guidance_context is not None
            and guidance_context.get("available")
            and guidance_context.get("ready_for_align")
        )

    def _guidance_allows_guarded_insert(self, guidance_context) -> bool:
        if not self.privileged_terminal_target_enabled:
            return True
        allowed = bool(
            guidance_context is not None
            and guidance_context.get("available")
            and guidance_context.get("ready_for_insert")
        )
        if allowed and guidance_context is not None:
            insert_axis = guidance_context.get("insert_direction_world")
            if isinstance(insert_axis, np.ndarray):
                self._guard_direction = insert_axis.copy()
        return allowed

    def _guidance_retry_phase(self, guidance_context) -> str:
        if self.privileged_terminal_target_enabled:
            return "ALIGN"
        return "ALIGN"

    def _guidance_summary(self, guidance_context) -> str:
        if guidance_context is None or not guidance_context.get("available"):
            return ""
        xy = guidance_context.get("plug_port_xy_error_m")
        dist = guidance_context.get("plug_port_distance_m")
        depth = guidance_context.get("insertion_depth_m")
        frac = guidance_context.get("insertion_fraction")
        ready = guidance_context.get("ready_for_insert")
        return (
            f"gt_xy={xy:.4f}m gt_dist={dist:.4f}m "
            f"gt_depth={depth:.4f}m gt_frac={frac:.3f} gt_ready={ready}"
        )

    def _lookup_tf_transform(self, target_frame: str, source_frame: str):
        return self._parent_node._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
        ).transform

    @staticmethod
    def _transform_translation_array(transform) -> np.ndarray:
        return np.array(
            [
                float(transform.translation.x),
                float(transform.translation.y),
                float(transform.translation.z),
            ],
            dtype=np.float64,
        )

    def _privileged_gripper_pose_for_offset(
        self,
        *,
        port_transform,
        plug_transform,
        gripper_transform,
        z_offset_m: float,
        target_xy_offset_m: np.ndarray,
    ) -> Pose:
        q_port = _quat_xyzw_to_array(port_transform.rotation)
        q_plug = _quat_xyzw_to_array(plug_transform.rotation)
        q_gripper = _quat_xyzw_to_array(gripper_transform.rotation)
        q_diff = _quat_xyzw_multiply(q_port, _quat_xyzw_inverse(q_plug))
        q_gripper_target = _quat_xyzw_multiply(q_diff, q_gripper)

        port_xyz = self._transform_translation_array(port_transform)
        plug_xyz = self._transform_translation_array(plug_transform)
        gripper_xyz = self._transform_translation_array(gripper_transform)
        plug_tip_gripper_offset = gripper_xyz - plug_xyz
        target_xy = port_xyz[:2] + target_xy_offset_m
        target_z = port_xyz[2] + float(z_offset_m) - plug_tip_gripper_offset[2]
        return Pose(
            position=Point(
                x=float(target_xy[0]),
                y=float(target_xy[1]),
                z=float(target_z),
            ),
            orientation=_quat_xyzw_to_msg(q_gripper_target),
        )

    def _build_privileged_terminal_context(self, observation_msg, task: Task):
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        port_entrance_frame = f"{port_frame}_entrance"
        plug_frame = f"{task.cable_name}/{task.plug_name}_link"
        gripper_frame = os.environ.get(
            "AIC_ACT_STAGED_PRIVILEGED_GRIPPER_FRAME", "gripper/tcp"
        )
        try:
            port_tf = self._lookup_tf_transform("base_link", port_frame)
            port_entrance_tf = self._lookup_tf_transform(
                "base_link", port_entrance_frame
            )
            plug_tf = self._lookup_tf_transform("base_link", plug_frame)
            gripper_tf = self._lookup_tf_transform("base_link", gripper_frame)
        except TransformException as ex:
            if self.geometry_debug_enabled or self.privileged_terminal_target_enabled:
                self.get_logger().warning(
                    f"Privileged terminal geometry unavailable: {ex}"
                )
            return {"available": False, "error": str(ex)}

        port_xyz = self._transform_translation_array(port_tf)
        entrance_xyz = self._transform_translation_array(port_entrance_tf)
        plug_xyz = self._transform_translation_array(plug_tf)
        gripper_xyz = self._transform_translation_array(gripper_tf)
        plug_port_delta = plug_xyz - port_xyz
        tip_xy_error = port_xyz[:2] - plug_xyz[:2]
        self._privileged_tip_xy_error_integrator = np.clip(
            self._privileged_tip_xy_error_integrator + tip_xy_error,
            -self.privileged_xy_integrator_windup_m,
            self.privileged_xy_integrator_windup_m,
        )
        target_xy_offset_m = (
            self.privileged_xy_direct_gain * tip_xy_error
            + self.privileged_xy_integrator_gain
            * self._privileged_tip_xy_error_integrator
        )
        xy_error = float(np.linalg.norm(plug_port_delta[:2]))
        distance = float(np.linalg.norm(plug_port_delta))
        port_depth = max(float(entrance_xyz[2] - port_xyz[2]), 1.0e-6)
        insertion_depth = float(entrance_xyz[2] - plug_xyz[2])
        insertion_fraction = float(insertion_depth / port_depth)
        hover_pose = self._privileged_gripper_pose_for_offset(
            port_transform=port_tf,
            plug_transform=plug_tf,
            gripper_transform=gripper_tf,
            z_offset_m=self.privileged_hover_offset_m,
            target_xy_offset_m=target_xy_offset_m,
        )
        align_pose = self._privileged_gripper_pose_for_offset(
            port_transform=port_tf,
            plug_transform=plug_tf,
            gripper_transform=gripper_tf,
            z_offset_m=self.privileged_align_offset_m,
            target_xy_offset_m=target_xy_offset_m,
        )
        insert_pose = self._privileged_gripper_pose_for_offset(
            port_transform=port_tf,
            plug_transform=plug_tf,
            gripper_transform=gripper_tf,
            z_offset_m=self.privileged_insert_target_offset_m,
            target_xy_offset_m=target_xy_offset_m,
        )
        insert_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        ready_for_align = xy_error <= max(0.025, self.privileged_ready_xy_tol_m * 4.0)
        ready_for_insert = xy_error <= self.privileged_ready_xy_tol_m
        current_pose = self._current_tcp_pose(observation_msg)
        current_xyz = self._pose_position_array(current_pose)
        guard_target_xyz = self._pose_position_array(insert_pose)
        raw_lateral = guard_target_xyz - current_xyz
        raw_lateral = raw_lateral - float(np.dot(raw_lateral, insert_axis)) * insert_axis
        context = {
            "available": True,
            "vision_mode": "privileged_terminal_target",
            "advisory_only": not self.privileged_terminal_target_enabled,
            "guidance_applied": bool(self.privileged_terminal_target_enabled),
            "guidance_would_apply": True,
            "ready_for_align": ready_for_align,
            "ready_for_insert": ready_for_insert,
            "port_frame": port_frame,
            "port_entrance_frame": port_entrance_frame,
            "plug_frame": plug_frame,
            "gripper_frame": gripper_frame,
            "port_position": port_xyz.astype(float).tolist(),
            "port_entrance_position": entrance_xyz.astype(float).tolist(),
            "plug_position": plug_xyz.astype(float).tolist(),
            "gripper_position": gripper_xyz.astype(float).tolist(),
            "plug_port_delta_m": plug_port_delta.astype(float).tolist(),
            "privileged_tip_xy_error_integrator_m": self._privileged_tip_xy_error_integrator.astype(float).tolist(),
            "privileged_target_xy_offset_m": target_xy_offset_m.astype(float).tolist(),
            "plug_port_xy_error_m": xy_error,
            "plug_port_distance_m": distance,
            "insertion_depth_m": insertion_depth,
            "insertion_fraction": insertion_fraction,
            "hover_pose": hover_pose,
            "align_pose": align_pose,
            "insert_pose": insert_pose,
            "insert_direction_world": insert_axis,
            "guard_lateral_correction_world_m": (
                self.privileged_guard_lateral_gain * raw_lateral
            ).astype(float).tolist(),
            "proposed_local_offset_m": plug_port_delta.astype(float).tolist(),
            "proposed_world_offset_m": plug_port_delta.astype(float).tolist(),
        }
        self._log_privileged_terminal_context(context)
        return context

    def _log_privileged_terminal_context(self, context: dict[str, object]) -> None:
        if not (self.geometry_debug_enabled or self.privileged_terminal_target_enabled):
            return
        every = max(self.privileged_debug_every_n, 1)
        self._privileged_debug_counter += 1
        phase = str(getattr(self, "_phase", "unknown"))
        important_phase = phase in {"ALIGN", "GUARDED_INSERT", "RETRY_OR_FINISH"}
        if self._privileged_debug_counter % every != 0 and not important_phase:
            return
        self.get_logger().info(
            "privileged_terminal_geometry "
            f"phase={phase} "
            f"xy_error_m={context['plug_port_xy_error_m']:.5f} "
            f"distance_m={context['plug_port_distance_m']:.5f} "
            f"depth_m={context['insertion_depth_m']:.5f} "
            f"fraction={context['insertion_fraction']:.3f} "
            f"ready_align={context['ready_for_align']} "
            f"ready_insert={context['ready_for_insert']} "
            f"target_enabled={self.privileged_terminal_target_enabled}"
        )

    def _reset_privileged_terminal_guidance(self) -> None:
        self._last_guidance_context = None
        self._privileged_debug_counter = 0
        self._privileged_tip_xy_error_integrator = np.zeros(2, dtype=np.float64)

    def _guidance_result_fields(self, guidance_context) -> dict[str, object]:
        if guidance_context is None or not guidance_context.get("available"):
            return {
                "gt_geometry_available": False,
                "gt_plug_port_xy_error_m": None,
                "gt_plug_port_distance_m": None,
                "gt_insertion_depth_m": None,
                "gt_insertion_fraction": None,
                "gt_ready_for_align": None,
                "gt_ready_for_insert": None,
                "gt_terminal_target_enabled": bool(
                    self.privileged_terminal_target_enabled
                ),
            }
        return {
            "gt_geometry_available": True,
            "gt_plug_port_xy_error_m": guidance_context.get("plug_port_xy_error_m"),
            "gt_plug_port_distance_m": guidance_context.get("plug_port_distance_m"),
            "gt_insertion_depth_m": guidance_context.get("insertion_depth_m"),
            "gt_insertion_fraction": guidance_context.get("insertion_fraction"),
            "gt_ready_for_align": guidance_context.get("ready_for_align"),
            "gt_ready_for_insert": guidance_context.get("ready_for_insert"),
            "gt_terminal_target_enabled": bool(self.privileged_terminal_target_enabled),
        }

    def _privileged_guard_pose_adjustment(
        self,
        proposed_pose: Pose,
        observation_msg,
        task: Task | None,
    ) -> Pose:
        if (
            not self.privileged_terminal_target_enabled
            or task is None
            or not self.enable_staged_wrapper
        ):
            return proposed_pose
        guidance_context = self._build_guidance_context(observation_msg, task)
        if guidance_context is None or not guidance_context.get("available"):
            return proposed_pose
        insert_axis = guidance_context.get("insert_direction_world")
        lateral = guidance_context.get("guard_lateral_correction_world_m")
        insert_pose = guidance_context.get("insert_pose")
        if not isinstance(insert_axis, np.ndarray) or not isinstance(lateral, list):
            return proposed_pose
        adjusted = self._copy_pose(proposed_pose)
        current_pos = self._pose_position_array(self._current_tcp_pose(observation_msg))
        proposed_pos = self._pose_position_array(proposed_pose)
        axial_delta = float(np.dot(proposed_pos - current_pos, insert_axis))
        lateral_vec = np.asarray(lateral, dtype=np.float64)
        adjusted_pos = current_pos + axial_delta * insert_axis + lateral_vec
        adjusted.position = Point(
            x=float(adjusted_pos[0]),
            y=float(adjusted_pos[1]),
            z=float(adjusted_pos[2]),
        )
        if isinstance(insert_pose, Pose):
            adjusted.orientation = insert_pose.orientation
        return adjusted
        return adjusted

    def _resolve_stage_policy_route(
        self,
        *,
        phase: str,
        env_var: str,
        default_route: str,
    ) -> str | None:
        configured = self._normalize_route_text(os.environ.get(env_var))
        if configured:
            return configured
        if default_route in self._policy_bundles:
            return default_route
        return None

    def _activate_phase_policy_bundle(
        self, phase: str, reason: str = ""
    ) -> str:
        route_key = self.stage_policy_routes.get(phase)
        if not route_key:
            return self._active_policy_route
        if route_key not in self._policy_bundles:
            self.get_logger().warning(
                f"Stage route {route_key!r} for phase {phase} is not loaded; "
                f"continuing with route={self._active_policy_route!r}"
            )
            return self._active_policy_route
        if route_key != self._active_policy_route:
            self._activate_policy_bundle(
                route_key, reason=f"phase={phase} {reason}".strip()
            )
            self.policy.reset()
        return self._active_policy_route

    def _set_phase(self, phase: str, now: float, reason: str = "") -> None:
        if self._phase == phase:
            return
        self._phase = phase
        self._phase_start_time = now
        self._recent_act_pose_positions.clear()
        if phase != "ALIGN":
            self._align_stable_since = None
        if phase != "GUARDED_INSERT":
            self._guard_start_pos = None
            self._best_guard_progress_m = 0.0
        if phase != "RETRY_OR_FINISH":
            self._retry_target_pose = None
            self._retry_reason = ""
        self._activate_phase_policy_bundle(phase, reason=reason)
        suffix = f" ({reason})" if reason else ""
        self.get_logger().info(f"RunACTStaged phase -> {phase}{suffix}")

    def _record_act_pose_target(self, pose: Pose) -> None:
        self._recent_act_pose_positions.append(self._pose_position_array(pose))

    def _act_target_is_stable(self) -> bool:
        if len(self._recent_act_pose_positions) < self._recent_act_pose_positions.maxlen:
            return False
        positions = np.stack(tuple(self._recent_act_pose_positions), axis=0)
        span = np.max(positions, axis=0) - np.min(positions, axis=0)
        return float(np.linalg.norm(span)) <= self.align_position_tolerance_m

    def _current_tcp_pose(self, observation_msg) -> Pose:
        return observation_msg.controller_state.tcp_pose

    def _tcp_error_metrics(self, observation_msg) -> tuple[float, float, float]:
        tcp_error = observation_msg.controller_state.tcp_error
        position_error = self._norm(tcp_error[:3])
        xy_error = self._norm(tcp_error[:2])
        angular_error = self._norm(tcp_error[3:6])
        return position_error, xy_error, angular_error

    def _apply_position_offset(self, pose: Pose, offset_xyz: np.ndarray) -> Pose:
        offset_pose = self._copy_pose(pose)
        offset_pose.position.x += float(offset_xyz[0])
        offset_pose.position.y += float(offset_xyz[1])
        offset_pose.position.z += float(offset_xyz[2])
        return offset_pose

    def _interpolate_pose(
        self,
        current_pose: Pose,
        target_pose: Pose,
        position_scale: float,
    ) -> Pose:
        current = self._pose_position_array(current_pose)
        target = self._pose_position_array(target_pose)
        alpha = float(np.clip(position_scale, 0.0, 1.0))
        shaped = current + alpha * (target - current)
        pose = self._copy_pose(target_pose)
        pose.position = Point(
            x=float(shaped[0]),
            y=float(shaped[1]),
            z=float(shaped[2]),
        )
        return pose

    def _estimate_insert_direction(
        self,
        observation_msg,
        act_pose: Pose,
    ) -> np.ndarray:
        if self.privileged_terminal_target_enabled:
            guidance_context = self._last_guidance_context
            if guidance_context is not None:
                insert_axis = guidance_context.get("insert_direction_world")
                if isinstance(insert_axis, np.ndarray):
                    return insert_axis.copy()
        current_pos = self._pose_position_array(self._current_tcp_pose(observation_msg))
        target_pos = self._pose_position_array(act_pose)
        target_delta = target_pos - current_pos
        if float(np.linalg.norm(target_delta)) > 2.0e-3:
            return self._normalized(target_delta, np.array([0.0, 0.0, -1.0]))
        if len(self._recent_act_pose_positions) >= 2:
            recent_delta = self._recent_act_pose_positions[-1] - self._recent_act_pose_positions[0]
            if float(np.linalg.norm(recent_delta)) > 1.0e-3:
                return self._normalized(recent_delta, np.array([0.0, 0.0, -1.0]))
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _enter_retry(
        self,
        observation_msg,
        insert_direction: np.ndarray,
        now: float,
        reason: str,
    ) -> None:
        current_pose = self._current_tcp_pose(observation_msg)
        current_pos = self._pose_position_array(current_pose)
        backout_pos = current_pos - insert_direction * self.backout_distance_m
        self._retry_target_pose = Pose(
            position=Point(
                x=float(backout_pos[0]),
                y=float(backout_pos[1]),
                z=float(backout_pos[2]),
            ),
            orientation=current_pose.orientation,
        )
        self._retry_reason = reason
        self._retries_used += 1
        self._set_phase("RETRY_OR_FINISH", now, reason=reason)

    def _maybe_finish_without_event(self, reason: str) -> bool:
        self.get_logger().info(
            "RunACTStaged finishing without insertion event. "
            f"reason={reason} require_insertion_event={self.require_insertion_event} "
            f"max_force_seen_n={self._max_force_seen_n:.4f}"
        )
        return True if not self.require_insertion_event else self._insertion_event_seen

    def step(self, action: np.ndarray) -> dict[str, object]:
        if self._active_observation_msg is None:
            raise RuntimeError("No active observation is available for wrapper step().")
        if self._active_task is None:
            raise RuntimeError("No active task is available for wrapper step().")
        if self._active_move_robot is None:
            raise RuntimeError("No active move_robot callback is available for step().")
        if action.shape[0] < 6:
            raise RuntimeError(
                f"ACT action has unexpected shape {action.shape}; need at least 6 dims"
            )

        observation_msg = self._active_observation_msg
        task = self._active_task
        move_robot = self._active_move_robot
        guidance_context = self._build_guidance_context(observation_msg, task)
        guidance_summary = self._guidance_summary(guidance_context)
        timestamp = self._clock_now_seconds()
        action_id = self._episode_action_id
        self._episode_action_id += 1
        phase_before = self._phase
        phase_now = self._phase
        loop_now = time.time()
        phase_elapsed = loop_now - self._phase_start_time

        compensated_force_n = self._compensated_force_magnitude(observation_msg)
        if compensated_force_n is not None:
            self._max_force_seen_n = max(self._max_force_seen_n, compensated_force_n)

        pose_position = action[6:9] if action.shape[0] >= 9 else np.zeros(3)
        pose_quaternion = None
        joint_targets = action[13:19] if action.shape[0] >= 19 else np.zeros(6)
        shaped_linear_log = None
        shaped_angular_log = None
        shaped_pose_position_log = [float("nan")] * 3
        wrench_feedback_gains = self.base_wrench_feedback_gains.copy()
        action_accepted = False
        action_type = "cartesian_delta"
        termination_reason = None
        wrapper_returned_success = None
        is_recovery = phase_now == "RETRY_OR_FINISH"

        if not self.enable_staged_wrapper:
            if self.command_mode == "pose" and action.shape[0] >= 13:
                quat = np.array(action[9:13], dtype=np.float64)
                quat_norm = float(np.linalg.norm(quat))
                quat = (
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    if quat_norm < 1e-8
                    else quat / quat_norm
                )
                pose_quaternion = quat.astype(float).tolist()
                pose = Pose(
                    position=Point(
                        x=float(action[6]),
                        y=float(action[7]),
                        z=float(action[8]),
                    ),
                    orientation=Quaternion(
                        x=float(quat[0]),
                        y=float(quat[1]),
                        z=float(quat[2]),
                        w=float(quat[3]),
                    ),
                )
                shaped_pose, wrench_feedback_gains, compensated_force_n = self._shape_pose(
                    pose, observation_msg
                )
                shaped_pose_position_log = [
                    float(shaped_pose.position.x),
                    float(shaped_pose.position.y),
                    float(shaped_pose.position.z),
                ]
                action_accepted = bool(
                    move_robot(
                        motion_update=self.create_pose_motion_update(
                            shaped_pose,
                            wrench_feedback_gains=wrench_feedback_gains,
                        )
                    )
                )
                action_type = "pose_target"
            else:
                twist = Twist(
                    linear=Vector3(
                        x=float(action[0]), y=float(action[1]), z=float(action[2])
                    ),
                    angular=Vector3(
                        x=float(action[3]), y=float(action[4]), z=float(action[5])
                    ),
                )
                shaped_twist, wrench_feedback_gains, compensated_force_n = (
                    self._shape_twist(twist, observation_msg)
                )
                shaped_linear_log = [
                    float(shaped_twist.linear.x),
                    float(shaped_twist.linear.y),
                    float(shaped_twist.linear.z),
                ]
                shaped_angular_log = [
                    float(shaped_twist.angular.x),
                    float(shaped_twist.angular.y),
                    float(shaped_twist.angular.z),
                ]
                action_accepted = bool(
                    move_robot(
                        motion_update=self.set_cartesian_twist_target(
                            shaped_twist,
                            wrench_feedback_gains=wrench_feedback_gains,
                        )
                    )
                )
        else:
            if self.command_mode != "pose" or action.shape[0] < 13:
                twist = Twist(
                    linear=Vector3(
                        x=float(action[0]), y=float(action[1]), z=float(action[2])
                    ),
                    angular=Vector3(
                        x=float(action[3]), y=float(action[4]), z=float(action[5])
                    ),
                )
                shaped_twist, wrench_feedback_gains, compensated_force_n = (
                    self._shape_twist(twist, observation_msg)
                )
                shaped_linear_log = [
                    float(shaped_twist.linear.x),
                    float(shaped_twist.linear.y),
                    float(shaped_twist.linear.z),
                ]
                shaped_angular_log = [
                    float(shaped_twist.angular.x),
                    float(shaped_twist.angular.y),
                    float(shaped_twist.angular.z),
                ]
                action_accepted = bool(
                    move_robot(
                        motion_update=self.set_cartesian_twist_target(
                            shaped_twist,
                            wrench_feedback_gains=wrench_feedback_gains,
                        )
                    )
                )
            else:
                quat = np.array(action[9:13], dtype=np.float64)
                quat_norm = float(np.linalg.norm(quat))
                quat = (
                    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                    if quat_norm < 1e-8
                    else quat / quat_norm
                )
                pose_quaternion = quat.astype(float).tolist()
                act_pose = Pose(
                    position=Point(
                        x=float(action[6]),
                        y=float(action[7]),
                        z=float(action[8]),
                    ),
                    orientation=Quaternion(
                        x=float(quat[0]),
                        y=float(quat[1]),
                        z=float(quat[2]),
                        w=float(quat[3]),
                    ),
                )
                self._record_act_pose_target(act_pose)
                current_pose = self._current_tcp_pose(observation_msg)
                current_pos = self._pose_position_array(current_pose)
                position_error, xy_error, angular_error = self._tcp_error_metrics(
                    observation_msg
                )
                action_type = "pose_target"

                if phase_now == "APPROACH":
                    approach_pose = self._apply_position_offset(
                        act_pose, self.approach_offset
                    )
                    approach_pose = self._apply_guidance_pose(
                        approach_pose,
                        observation_msg,
                        task,
                        guidance_context,
                    )
                    shaped_pose, wrench_feedback_gains, compensated_force_n = self._shape_pose(
                        approach_pose, observation_msg
                    )
                    shaped_pose_position_log = [
                        float(shaped_pose.position.x),
                        float(shaped_pose.position.y),
                        float(shaped_pose.position.z),
                    ]
                    action_accepted = bool(
                        move_robot(
                            motion_update=self.create_pose_motion_update(
                                shaped_pose,
                                wrench_feedback_gains=wrench_feedback_gains,
                            )
                        )
                    )
                    if (
                        phase_elapsed >= self.approach_min_sec
                        and self._act_target_is_stable()
                        and position_error <= self.align_position_tolerance_m
                        and self._guidance_allows_align(guidance_context)
                    ):
                        self._set_phase("ALIGN", loop_now, reason="target_stable")

                elif phase_now == "ALIGN":
                    approach_pose = self._apply_position_offset(
                        act_pose, self.approach_offset
                    )
                    approach_pose = self._apply_guidance_pose(
                        approach_pose,
                        observation_msg,
                        task,
                        guidance_context,
                    )
                    align_pose = self._interpolate_pose(
                        current_pose,
                        approach_pose,
                        self.guard_position_smoothing_scale,
                    )
                    shaped_pose, wrench_feedback_gains, compensated_force_n = self._shape_pose(
                        align_pose, observation_msg
                    )
                    shaped_pose_position_log = [
                        float(shaped_pose.position.x),
                        float(shaped_pose.position.y),
                        float(shaped_pose.position.z),
                    ]
                    action_accepted = bool(
                        move_robot(
                            motion_update=self.create_pose_motion_update(
                                shaped_pose,
                                wrench_feedback_gains=wrench_feedback_gains,
                            )
                        )
                    )
                    align_ok = (
                        self._act_target_is_stable()
                        and xy_error <= self.align_xy_error_tolerance_m
                        and angular_error <= self.align_angular_tolerance_rad
                        and self._guidance_allows_guarded_insert(guidance_context)
                    )
                    if align_ok:
                        if self._align_stable_since is None:
                            self._align_stable_since = loop_now
                        elif loop_now - self._align_stable_since >= self.settle_time_s:
                            self._guard_direction = self._estimate_insert_direction(
                                observation_msg, act_pose
                            )
                            self._guard_start_pos = current_pos.copy()
                            self._best_guard_progress_m = 0.0
                            self._set_phase(
                                "GUARDED_INSERT",
                                loop_now,
                                reason="aligned",
                            )
                    else:
                        self._align_stable_since = None

                elif phase_now == "GUARDED_INSERT":
                    if self._guard_direction is None:
                        self._guard_direction = self._estimate_insert_direction(
                            observation_msg, act_pose
                        )
                    if self._guard_start_pos is None:
                        self._guard_start_pos = current_pos.copy()

                    step_m = max(
                        0.001,
                        self.max_linear_speed_mps
                        * self.control_period_s
                        * self.insertion_speed_scale,
                    )
                    if (
                        compensated_force_n is not None
                        and compensated_force_n > self.guarded_insert_force_start_n
                    ):
                        ramp_span = max(
                            self.guarded_insert_force_stop_n
                            - self.guarded_insert_force_start_n,
                            1e-6,
                        )
                        ramp_scale = 1.0 - (
                            (
                                compensated_force_n
                                - self.guarded_insert_force_start_n
                            )
                            / ramp_span
                        )
                        step_m *= float(np.clip(ramp_scale, 0.1, 1.0))

                    guard_target_pos = current_pos + self._guard_direction * step_m
                    guard_pose = Pose(
                        position=Point(
                            x=float(guard_target_pos[0]),
                            y=float(guard_target_pos[1]),
                            z=float(guard_target_pos[2]),
                        ),
                        orientation=act_pose.orientation,
                    )
                    guard_pose = self._privileged_guard_pose_adjustment(
                        guard_pose,
                        observation_msg,
                        task,
                    )
                    shaped_pose, wrench_feedback_gains, compensated_force_n = self._shape_pose(
                        guard_pose, observation_msg
                    )
                    shaped_pose_position_log = [
                        float(shaped_pose.position.x),
                        float(shaped_pose.position.y),
                        float(shaped_pose.position.z),
                    ]
                    action_accepted = bool(
                        move_robot(
                            motion_update=self.create_pose_motion_update(
                                shaped_pose,
                                wrench_feedback_gains=wrench_feedback_gains,
                            )
                        )
                    )
                    progress_m = float(
                        np.dot(
                            current_pos - self._guard_start_pos,
                            self._guard_direction,
                        )
                    )
                    self._best_guard_progress_m = max(
                        self._best_guard_progress_m, progress_m
                    )
                    if (
                        compensated_force_n is not None
                        and compensated_force_n >= self.guarded_insert_force_stop_n
                    ):
                        if self._retries_used < self.max_retries:
                            self._enter_retry(
                                observation_msg,
                                self._guard_direction,
                                loop_now,
                                reason="force_stop",
                            )
                        else:
                            termination_reason = "force_stop"
                            wrapper_returned_success = self._maybe_finish_without_event(
                                "force_stop"
                            )
                    elif phase_elapsed >= self.max_guarded_insert_sec:
                        if self._best_guard_progress_m < self.guard_progress_epsilon_m:
                            if self._retries_used < self.max_retries:
                                self._enter_retry(
                                    observation_msg,
                                    self._guard_direction,
                                    loop_now,
                                    reason="no_progress",
                                )
                            else:
                                termination_reason = "no_progress"
                                wrapper_returned_success = self._maybe_finish_without_event(
                                    "no_progress"
                                )
                        else:
                            termination_reason = "guarded_insert_complete"
                            wrapper_returned_success = self._maybe_finish_without_event(
                                "guarded_insert_complete"
                            )

                elif phase_now == "RETRY_OR_FINISH":
                    if self._retry_target_pose is None:
                        termination_reason = "retry_missing_target"
                        wrapper_returned_success = self._maybe_finish_without_event(
                            "retry_missing_target"
                        )
                    else:
                        shaped_pose, wrench_feedback_gains, compensated_force_n = self._shape_pose(
                            self._retry_target_pose, observation_msg
                        )
                        shaped_pose_position_log = [
                            float(shaped_pose.position.x),
                            float(shaped_pose.position.y),
                            float(shaped_pose.position.z),
                        ]
                        action_accepted = bool(
                            move_robot(
                                motion_update=self.create_pose_motion_update(
                                    shaped_pose,
                                    wrench_feedback_gains=wrench_feedback_gains,
                                )
                            )
                        )
                        if phase_elapsed >= self.settle_time_s:
                            if self._retries_used <= self.max_retries:
                                retry_phase = self._guidance_retry_phase(
                                    guidance_context
                                )
                                self._set_phase(
                                    retry_phase,
                                    loop_now,
                                    reason=f"retry_{self._retry_reason}",
                                )
                            else:
                                termination_reason = "retry_budget_exhausted"
                                wrapper_returned_success = self._maybe_finish_without_event(
                                    "retry_budget_exhausted"
                                )

        if not action_accepted and termination_reason is None:
            termination_reason = "move_robot_rejected"
            wrapper_returned_success = False

        shaped_pose_valid = not any(np.isnan(value) for value in shaped_pose_position_log)
        if action_type == "pose_target":
            action_clipped = shaped_pose_valid and not np.allclose(
                np.asarray(pose_position, dtype=np.float64),
                np.asarray(shaped_pose_position_log, dtype=np.float64),
                atol=1e-6,
            )
        else:
            raw_linear = np.asarray(action[:3], dtype=np.float64)
            raw_angular = np.asarray(action[3:6], dtype=np.float64)
            action_clipped = (
                shaped_linear_log is not None
                and not np.allclose(
                    raw_linear,
                    np.asarray(shaped_linear_log, dtype=np.float64),
                    atol=1e-6,
                )
            ) or (
                shaped_angular_log is not None
                and not np.allclose(
                    raw_angular,
                    np.asarray(shaped_angular_log, dtype=np.float64),
                    atol=1e-6,
                )
            )

        feedback = f"phase={self._phase}"
        if compensated_force_n is not None:
            feedback += f" force={compensated_force_n:.2f}N"
        if self._phase == "RETRY_OR_FINISH":
            feedback += f" retry_reason={self._retry_reason}"
        if guidance_summary:
            feedback += f" {guidance_summary}"

        guidance_mode = None
        guidance_advisory_only = None
        guidance_applied = None
        guidance_would_apply = None
        guidance_ready_for_align = None
        guidance_ready_for_insert = None
        guidance_proposed_local_offset_m = None
        guidance_proposed_world_offset_m = None
        if guidance_context is not None:
            guidance_mode = guidance_context.get("vision_mode")
            guidance_advisory_only = guidance_context.get("advisory_only")
            guidance_applied = guidance_context.get("guidance_applied")
            guidance_would_apply = guidance_context.get("guidance_would_apply")
            guidance_ready_for_align = guidance_context.get("ready_for_align")
            guidance_ready_for_insert = guidance_context.get("ready_for_insert")
            guidance_proposed_local_offset_m = guidance_context.get(
                "proposed_local_offset_m"
            )
            guidance_proposed_world_offset_m = guidance_context.get(
                "proposed_world_offset_m"
            )

        result = {
            "timestamp": timestamp,
            "action_id": action_id,
            "source": self._episode_source,
            "policy_route": self._active_policy_route,
            "action_type": action_type,
            "raw_action": action.astype(float).tolist(),
            "raw_linear": action[:3].astype(float).tolist(),
            "raw_angular": action[3:6].astype(float).tolist(),
            "pose_target_position": pose_position.astype(float).tolist()
            if action.shape[0] >= 9
            else None,
            "pose_target_quaternion": pose_quaternion,
            "shaped_linear": shaped_linear_log,
            "shaped_angular": shaped_angular_log,
            "shaped_pose_position": shaped_pose_position_log,
            "joint_target_norm": float(np.linalg.norm(joint_targets)),
            "wrench_feedback_gains": list(wrench_feedback_gains),
            "force_n": compensated_force_n,
            "max_force_n": self._max_force_seen_n,
            "action_accepted": action_accepted,
            "action_clipped": bool(action_clipped),
            "phase_before": phase_before,
            "phase_after": self._phase,
            "termination_reason": termination_reason,
            "wrapper_returned_success": wrapper_returned_success,
            "is_recovery": is_recovery,
            "guidance_summary": guidance_summary,
            "guidance_mode": guidance_mode,
            "guidance_advisory_only": guidance_advisory_only,
            "guidance_applied": guidance_applied,
            "guidance_would_apply": guidance_would_apply,
            "guidance_ready_for_align": guidance_ready_for_align,
            "guidance_ready_for_insert": guidance_ready_for_insert,
            "guidance_proposed_local_offset_m": guidance_proposed_local_offset_m,
            "guidance_proposed_world_offset_m": guidance_proposed_world_offset_m,
            "feedback": feedback,
            "should_terminate": wrapper_returned_success is not None,
        }
        result.update(self._guidance_result_fields(guidance_context))
        return result

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self._activate_policy_bundle_for_task(task)
        self.policy.reset()
        self._insertion_event_seen = False
        self._last_insertion_event = ""
        self._recent_act_pose_positions.clear()
        self._phase = "APPROACH"
        self._phase_start_time = time.time()
        self._align_stable_since = None
        self._guard_direction = None
        self._guard_start_pos = None
        self._best_guard_progress_m = 0.0
        self._retry_target_pose = None
        self._retry_reason = ""
        self._retries_used = 0
        self._max_force_seen_n = 0.0
        self._reset_privileged_terminal_guidance()
        self._activate_phase_policy_bundle("APPROACH", reason="insert_start")
        self._begin_canonical_episode(task)
        self.get_logger().info(f"RunACTStaged.insert_cable() enter. Task: {task}")

        start_time = time.time()
        inference_count = 0

        while time.time() - start_time < 30.0:
            if self._insertion_event_seen:
                self.get_logger().info(
                    "Insertion event observed; exiting success. "
                    f"event={self._last_insertion_event!r} "
                    f"max_force_seen_n={self._max_force_seen_n:.4f}"
                )
                self._finalize_canonical_episode(
                    wrapper_returned_success=True,
                    termination_reason="insertion_event",
                )
                return True

            loop_start = time.time()
            observation_msg = get_observation()
            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            self._set_active_runtime_context(
                observation_msg=observation_msg,
                task_msg=task,
                move_robot=move_robot,
            )
            policy_observation = self.get_policy_observation()
            self._write_policy_observation(policy_observation)

            obs_tensors = self.prepare_observations(observation_msg, task)
            with torch.inference_mode():
                normalized_action = self.policy.select_action(obs_tensors)

            if not torch.isfinite(normalized_action).all():
                bad_count = int((~torch.isfinite(normalized_action)).sum().item())
                self.get_logger().warning(
                    f"Non-finite normalized ACT action detected: {bad_count}. Replacing with zeros."
                )
                normalized_action = torch.nan_to_num(normalized_action)

            raw_action_tensor = (normalized_action * self.action_std) + self.action_mean
            action = raw_action_tensor[0].detach().cpu().numpy()
            if not np.isfinite(action).all():
                bad_count = int(np.size(action) - np.isfinite(action).sum())
                self.get_logger().warning(
                    f"Non-finite unnormalized ACT action detected: {bad_count}. Replacing with zeros."
                )
                action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
            action_result = self.step(action)
            self._write_action_and_wrapper_output(
                observation=policy_observation,
                action_result=action_result,
            )

            if self.log_every_n_steps > 0 and inference_count % self.log_every_n_steps == 0:
                self.get_logger().info(
                    "ACT staged action "
                    f"step={inference_count + 1} "
                    f"phase={action_result.get('phase_after')} "
                    f"raw_linear={action_result.get('raw_linear')} "
                    f"raw_angular={action_result.get('raw_angular')} "
                    f"shaped_linear={action_result.get('shaped_linear') if action_result.get('shaped_linear') is not None else 'n/a'} "
                    f"shaped_angular={action_result.get('shaped_angular') if action_result.get('shaped_angular') is not None else 'n/a'} "
                    f"pose_position={action_result.get('pose_target_position')} "
                    f"shaped_pose_position={action_result.get('shaped_pose_position')} "
                    f"joint_norm={float(action_result.get('joint_target_norm', 0.0)):.6f} "
                    f"mode={self.command_mode} "
                    f"force_n={action_result.get('force_n') if action_result.get('force_n') is not None else float('nan'):.4f} "
                    f"max_force_n={self._max_force_seen_n:.4f} "
                    f"gains={action_result.get('wrench_feedback_gains')} "
                    f"retries_used={self._retries_used}/{self.max_retries} "
                    f"insertion_event={self._insertion_event_seen}"
                )

            send_feedback(str(action_result.get("feedback", f"phase={self._phase}")))
            if bool(action_result.get("should_terminate", False)):
                final_success = bool(action_result.get("wrapper_returned_success"))
                self._finalize_canonical_episode(
                    wrapper_returned_success=final_success,
                    termination_reason=str(action_result.get("termination_reason") or "terminated"),
                )
                return final_success
            inference_count += 1

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.control_period_s - elapsed))

        final_success = self._maybe_finish_without_event("timeout")
        self._finalize_canonical_episode(
            wrapper_returned_success=final_success,
            termination_reason="timeout",
        )
        return final_success
