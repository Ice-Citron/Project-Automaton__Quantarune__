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

import math
import os
import time

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .RunSCWaypointReplay import RunSCWaypointReplay, _float_env, _int_env
from .primitive_runtime import (
    CallbackPrimitive,
    PrimitiveContext,
    PrimitiveResult,
    PrimitiveRuntime,
)
from .guarded_insertion_recovery import (
    GuardedInsertionRecoveryConfig,
    GuardedInsertionRecoverySupervisor,
)
from .vision_target_features import CenterPortCVPipeline


class RunSCVisualServoReplay(RunSCWaypointReplay):
    """Template replay plus deployable image-based terminal centering.

    This policy is intentionally limited to legal runtime signals. It does not
    use /scoring/tf or ground-truth object poses. Instead, it probes tiny base
    frame x/y moves and learns the local camera error response online.
    """

    _DIRECT_METRIC_MAPPING_CANDIDATES = (
        ("minus_xy", (-1.0, 0.0, 0.0, -1.0)),
        ("plus_xy", (1.0, 0.0, 0.0, 1.0)),
        ("minus_x_plus_y", (-1.0, 0.0, 0.0, 1.0)),
        ("plus_x_minus_y", (1.0, 0.0, 0.0, -1.0)),
        ("swap_minus_y_minus_x", (0.0, -1.0, -1.0, 0.0)),
        ("swap_plus_y_plus_x", (0.0, 1.0, 1.0, 0.0)),
        ("swap_minus_y_plus_x", (0.0, -1.0, 1.0, 0.0)),
        ("swap_plus_y_minus_x", (0.0, 1.0, -1.0, 0.0)),
    )

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.hard_force_n = _float_env(
            "AIC_SC_VS_HARD_FORCE_N",
            min(self.hard_force_n, 19.0),
        )
        self.final_centimeter_profile = (
            os.environ.get(
                "AIC_SC_VS_FINAL_CM_PROFILE",
                os.environ.get(
                    "AIC_ACT_FINAL_CM_PROFILE",
                    os.environ.get("AIC_ACT_SC_DELTA16_FINAL_CM_PROFILE", "default"),
                ),
            )
            .strip()
            .lower()
        )
        gentle_final_cm = self.final_centimeter_profile in {
            "gentle",
            "gentle_commit",
            "teacher_gentle",
        }
        self.visual_start_waypoint = _int_env("AIC_SC_VS_START_WAYPOINT", 100)
        self.visual_start_fraction = _float_env("AIC_SC_VS_START_FRACTION", -1.0)
        self.probe_step_m = _float_env("AIC_SC_VS_PROBE_STEP_M", 0.0015)
        self.settle_s = _float_env("AIC_SC_VS_SETTLE_S", 0.18)
        self.max_center_steps = _int_env("AIC_SC_VS_MAX_CENTER_STEPS", 32)
        self.center_tol_px = _float_env("AIC_SC_VS_CENTER_TOL_PX", 22.0)
        self.center_tol_m = _float_env("AIC_SC_VS_CENTER_TOL_M", 0.0040)
        self.center_dwell_steps = _int_env("AIC_SC_VS_CENTER_DWELL_STEPS", 2)
        self.max_correction_step_m = _float_env("AIC_SC_VS_MAX_CORRECTION_STEP_M", 0.0030)
        self.max_correction_total_m = _float_env("AIC_SC_VS_MAX_CORRECTION_TOTAL_M", 0.020)
        self.max_metric_correction_total_m = _float_env(
            "AIC_SC_VS_MAX_METRIC_CORRECTION_TOTAL_M", max(self.max_correction_total_m, 0.060)
        )
        self.max_initial_error_px = _float_env("AIC_SC_VS_MAX_INITIAL_ERROR_PX", 420.0)
        self.max_initial_error_m = _float_env("AIC_SC_VS_MAX_INITIAL_ERROR_M", 0.150)
        self.jacobian_refresh_steps = _int_env("AIC_SC_VS_JACOBIAN_REFRESH_STEPS", 8)
        self.coarse_search_enabled = (
            os.environ.get("AIC_SC_VS_COARSE_SEARCH", "0").lower()
            not in {"0", "false", "no"}
        )
        self.coarse_search_radius_m = max(
            0.0, _float_env("AIC_SC_VS_COARSE_SEARCH_RADIUS_M", 0.040)
        )
        self.coarse_search_step_m = max(
            0.001, _float_env("AIC_SC_VS_COARSE_SEARCH_STEP_M", 0.010)
        )
        self.coarse_search_max_candidates = max(
            1, _int_env("AIC_SC_VS_COARSE_SEARCH_MAX_CANDIDATES", 49)
        )
        self.coarse_search_min_improvement_px = max(
            0.0, _float_env("AIC_SC_VS_COARSE_SEARCH_MIN_IMPROVEMENT_PX", 30.0)
        )
        self.coarse_search_min_improvement_m = max(
            0.0, _float_env("AIC_SC_VS_COARSE_SEARCH_MIN_IMPROVEMENT_M", 0.005)
        )
        self.coarse_search_accept_max_error_px = _float_env(
            "AIC_SC_VS_COARSE_SEARCH_ACCEPT_MAX_ERROR_PX",
            self.max_initial_error_px,
        )
        self.coarse_search_accept_max_error_m = _float_env(
            "AIC_SC_VS_COARSE_SEARCH_ACCEPT_MAX_ERROR_M",
            self.max_initial_error_m,
        )
        self.min_target_conf = _float_env("AIC_SC_VS_MIN_TARGET_CONF", 0.20)
        self.min_plug_conf = _float_env("AIC_SC_VS_MIN_PLUG_CONF", 0.15)
        self.use_pose_estimate = (
            os.environ.get("AIC_SC_VS_USE_POSE_ESTIMATE", "1").lower()
            not in {"0", "false", "no"}
        )
        self.pose_min_conf = _float_env("AIC_SC_VS_POSE_MIN_CONFIDENCE", 0.12)
        self.pose_max_reprojection_px = _float_env(
            "AIC_SC_VS_POSE_MAX_REPROJECTION_PX", 45.0
        )
        self.direct_metric_correction_enabled = (
            os.environ.get("AIC_SC_VS_DIRECT_METRIC_CORRECTION", "1").lower()
            not in {"0", "false", "no"}
        )
        self.direct_metric_gain = _float_env("AIC_SC_VS_DIRECT_METRIC_GAIN", 0.75)
        self.direct_metric_min_improvement_m = _float_env(
            "AIC_SC_VS_DIRECT_METRIC_MIN_IMPROVEMENT_M", 0.0010
        )
        self.direct_metric_mapping_override = _int_env(
            "AIC_SC_VS_DIRECT_METRIC_MAPPING_INDEX", -1
        )
        self.direct_metric_lock_after_accept = (
            os.environ.get("AIC_SC_VS_DIRECT_METRIC_LOCK_AFTER_ACCEPT", "1").lower()
            not in {"0", "false", "no"}
        )
        self.max_insert_steps = _int_env(
            "AIC_SC_VS_MAX_INSERT_STEPS",
            120 if gentle_final_cm else 70,
        )
        self.insert_step_m = _float_env(
            "AIC_SC_VS_INSERT_STEP_M",
            0.00025 if gentle_final_cm else 0.00055,
        )
        self.insert_axis_scale = _float_env("AIC_SC_VS_INSERT_AXIS_SCALE", 1.0)
        self.insert_axis_mode = (
            os.environ.get("AIC_SC_VS_INSERT_AXIS_MODE", "terminal").strip().lower()
        )
        self.terminal_axis_waypoints = max(
            1, _int_env("AIC_SC_VS_TERMINAL_AXIS_WAYPOINTS", 30)
        )
        self.terminal_pre_commit_hold_s = max(
            0.0, _float_env("AIC_SC_VS_PRE_COMMIT_HOLD_S", 0.35)
        )
        self.terminal_pre_commit_max_force_n = _float_env(
            "AIC_SC_VS_PRE_COMMIT_MAX_FORCE_N",
            min(self.hard_force_n, 18.0),
        )
        self.terminal_pre_commit_force_rise_n = max(
            0.0, _float_env("AIC_SC_VS_PRE_COMMIT_FORCE_RISE_N", 3.0)
        )
        self.terminal_pre_commit_visual_margin = max(
            1.0, _float_env("AIC_SC_VS_PRE_COMMIT_VISUAL_MARGIN", 1.75)
        )
        self.terminal_pre_commit_require_visual = (
            os.environ.get("AIC_SC_VS_PRE_COMMIT_REQUIRE_VISUAL", "0").lower()
            not in {"0", "false", "no"}
        )
        self.use_guarded_recovery = (
            os.environ.get("AIC_SC_VS_USE_GUARDED_RECOVERY", "1").lower()
            not in {"0", "false", "no"}
        )
        self.guarded_recovery_config = GuardedInsertionRecoveryConfig.from_env()
        if "AIC_ACT_GUARDED_RECOVERY_MODE" not in os.environ:
            self.guarded_recovery_config.mode = os.environ.get(
                "AIC_SC_VS_GUARDED_RECOVERY_MODE", "control"
            ).strip().lower()
        self.guarded_recovery_config.max_axial_insert_delta_m = _float_env(
            "AIC_SC_VS_GUARDED_MAX_AXIAL_DELTA_M",
            min(self.guarded_recovery_config.max_axial_insert_delta_m, self.insert_step_m),
        )
        self.guarded_recovery_config.max_lateral_delta_m = _float_env(
            "AIC_SC_VS_GUARDED_MAX_LATERAL_DELTA_M",
            self.guarded_recovery_config.max_lateral_delta_m,
        )
        self.guarded_recovery_config.soft_force_threshold_n = _float_env(
            "AIC_SC_VS_GUARDED_SOFT_FORCE_N",
            self.guarded_recovery_config.soft_force_threshold_n,
        )
        self.guarded_recovery_config.hard_force_threshold_n = _float_env(
            "AIC_SC_VS_GUARDED_HARD_FORCE_N",
            min(self.guarded_recovery_config.hard_force_threshold_n, self.hard_force_n),
        )
        self.guarded_recovery_config.insert_timeout_s = _float_env(
            "AIC_SC_VS_GUARDED_INSERT_TIMEOUT_S",
            max(self.guarded_recovery_config.insert_timeout_s, 8.0 if gentle_final_cm else 5.0),
        )
        self.guarded_recovery_config.stall_cycles = _int_env(
            "AIC_SC_VS_GUARDED_STALL_CYCLES",
            max(self.guarded_recovery_config.stall_cycles, 12 if gentle_final_cm else 8),
        )
        self.guarded_recovery_config.retry_budget = _int_env(
            "AIC_SC_VS_GUARDED_RETRY_BUDGET",
            max(self.guarded_recovery_config.retry_budget, 2),
        )
        self.guarded_recovery_config.backout_distance_m = _float_env(
            "AIC_SC_VS_GUARDED_BACKOUT_DISTANCE_M",
            self.guarded_recovery_config.backout_distance_m,
        )
        self.guarded_recovery = GuardedInsertionRecoverySupervisor(
            self.guarded_recovery_config
        )
        self.primitive_runtime_enabled = (
            os.environ.get("AIC_SC_VS_PRIMITIVE_RUNTIME", "1").lower()
            not in {"0", "false", "no"}
        )
        self.fallback_guarded_insert_enabled = (
            os.environ.get("AIC_SC_VS_FALLBACK_GUARDED_INSERT", "0").lower()
            not in {"0", "false", "no"}
        )
        self.carry_replay_offset = (
            os.environ.get("AIC_SC_VS_CARRY_REPLAY_OFFSET", "0").lower()
            not in {"0", "false", "no"}
        )
        self.primitive_runtime_history: list[dict[str, object]] = []
        self.cv_pipeline = CenterPortCVPipeline(camera_name="center")
        self._active_error_units = "pixels"
        self._direct_metric_mapping_index: int | None = None
        self.get_logger().info(
            "RunSCVisualServoReplay configured: "
            f"start_waypoint={self.visual_start_waypoint} "
            f"probe_step_m={self.probe_step_m:.4f} "
            f"center_tol_px={self.center_tol_px:.1f} "
            f"center_tol_m={self.center_tol_m:.4f} "
            f"use_pose_estimate={self.use_pose_estimate} "
            f"direct_metric_correction={self.direct_metric_correction_enabled} "
            f"direct_metric_gain={self.direct_metric_gain:.2f} "
            f"direct_metric_lock_after_accept={self.direct_metric_lock_after_accept} "
            f"max_initial_error_px={self.max_initial_error_px:.1f} "
            f"max_initial_error_m={self.max_initial_error_m:.3f} "
            f"coarse_search={self.coarse_search_enabled} "
            f"coarse_radius_m={self.coarse_search_radius_m:.3f} "
            f"coarse_step_m={self.coarse_search_step_m:.3f} "
            f"max_center_steps={self.max_center_steps} "
            f"final_cm_profile={self.final_centimeter_profile} "
            f"max_insert_steps={self.max_insert_steps} "
            f"insert_step_m={self.insert_step_m:.5f} "
            f"insert_axis_mode={self.insert_axis_mode} "
            f"pre_commit_hold_s={self.terminal_pre_commit_hold_s:.2f} "
            f"hard_force_n={self.hard_force_n:.1f} "
            f"guarded_recovery={self.use_guarded_recovery}:"
            f"{self.guarded_recovery_config.mode}:"
            f"timeout={self.guarded_recovery_config.insert_timeout_s:.1f}s:"
            f"hard={self.guarded_recovery_config.hard_force_threshold_n:.1f}N "
            f"primitive_runtime={self.primitive_runtime_enabled} "
            f"fallback_guarded_insert={self.fallback_guarded_insert_enabled} "
            f"carry_replay_offset={self.carry_replay_offset}"
        )

    @staticmethod
    def _copy_pose(pose: Pose) -> Pose:
        return Pose(
            position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            orientation=Quaternion(
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
                w=pose.orientation.w,
            ),
        )

    @staticmethod
    def _pose_from_observation(observation) -> Pose:
        pose = observation.controller_state.tcp_pose
        return Pose(
            position=Point(x=pose.position.x, y=pose.position.y, z=pose.position.z),
            orientation=Quaternion(
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
                w=pose.orientation.w,
            ),
        )

    @staticmethod
    def _offset_pose_xyz(pose: Pose, delta: np.ndarray) -> Pose:
        return Pose(
            position=Point(
                x=pose.position.x + float(delta[0]),
                y=pose.position.y + float(delta[1]),
                z=pose.position.z + float(delta[2]),
            ),
            orientation=Quaternion(
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
                w=pose.orientation.w,
            ),
        )

    @staticmethod
    def _pose_position_np(pose: Pose) -> np.ndarray:
        return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    def _force_from_observation(self, observation) -> float:
        wrench = observation.controller_state.fts_tare_offset.wrench.force
        obs_force = math.sqrt(wrench.x * wrench.x + wrench.y * wrench.y + wrench.z * wrench.z)
        self.max_force_magnitude_n = max(self.max_force_magnitude_n, obs_force)
        return max(self.last_force_magnitude_n, obs_force)

    def _error_tolerance(self, units: str) -> float:
        return self.center_tol_m if units == "meters" else self.center_tol_px

    def _max_initial_error(self, units: str) -> float:
        return self.max_initial_error_m if units == "meters" else self.max_initial_error_px

    def _coarse_min_improvement(self, units: str) -> float:
        return (
            self.coarse_search_min_improvement_m
            if units == "meters"
            else self.coarse_search_min_improvement_px
        )

    def _coarse_accept_max_error(self, units: str) -> float:
        return (
            self.coarse_search_accept_max_error_m
            if units == "meters"
            else self.coarse_search_accept_max_error_px
        )

    def _max_correction_total(self, units: str) -> float:
        return (
            self.max_metric_correction_total_m
            if units == "meters"
            else self.max_correction_total_m
        )

    @staticmethod
    def _format_error_norm(value: float, units: str) -> str:
        return f"{value:.4f}m" if units == "meters" else f"{value:.1f}px"

    def _metric_mapping_order(self) -> list[int]:
        count = len(self._DIRECT_METRIC_MAPPING_CANDIDATES)
        if 0 <= self.direct_metric_mapping_override < count:
            return [self.direct_metric_mapping_override]
        if self._direct_metric_mapping_index is not None:
            preferred = self._direct_metric_mapping_index % count
            if self.direct_metric_lock_after_accept:
                return [preferred]
            return [preferred] + [index for index in range(count) if index != preferred]
        return list(range(count))

    def _direct_metric_command_xy(
        self,
        *,
        error_m: np.ndarray,
        mapping_index: int,
    ) -> tuple[np.ndarray, str]:
        mapping_name, coeffs = self._DIRECT_METRIC_MAPPING_CANDIDATES[
            mapping_index % len(self._DIRECT_METRIC_MAPPING_CANDIDATES)
        ]
        a, b, c, d = coeffs
        command_xy = self.direct_metric_gain * np.array(
            [
                (a * float(error_m[0])) + (b * float(error_m[1])),
                (c * float(error_m[0])) + (d * float(error_m[1])),
            ],
            dtype=np.float64,
        )
        command_norm = float(np.linalg.norm(command_xy))
        if command_norm > self.max_correction_step_m and command_norm > 1e-9:
            command_xy *= self.max_correction_step_m / command_norm
        return command_xy, mapping_name

    def _apply_direct_metric_correction(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        current_pose: Pose,
        error_m: np.ndarray,
        total_correction: np.ndarray,
        max_correction_total: float,
        step: int,
    ) -> tuple[Pose, np.ndarray | None, str, np.ndarray | None]:
        start_norm = float(np.linalg.norm(error_m))
        best_seen_norm = start_norm
        best_seen_name = "none"

        for mapping_index in self._metric_mapping_order():
            correction_xy, mapping_name = self._direct_metric_command_xy(
                error_m=error_m,
                mapping_index=mapping_index,
            )
            correction_norm = float(np.linalg.norm(correction_xy))
            if correction_norm <= 1e-9:
                continue
            total_after = float(np.linalg.norm(total_correction + correction_xy))
            if total_after > max_correction_total:
                self.get_logger().warning(
                    "direct metric correction skipped budget "
                    f"mapping={mapping_name} total_after={total_after:.4f}m "
                    f"budget={max_correction_total:.4f}m"
                )
                continue

            target_pose = self._offset_pose_xyz(
                current_pose,
                np.array([correction_xy[0], correction_xy[1], 0.0], dtype=np.float64),
            )
            observation, post_error, post_status = self._move_and_observe(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                pose=target_pose,
                reason=(
                    "direct metric visual center "
                    f"step={step} mapping={mapping_name} "
                    f"cmd={correction_xy.astype(float).tolist()}"
                ),
            )
            if observation is not None:
                force_n = self._force_from_observation(observation)
                if force_n >= self.hard_force_n:
                    return (
                        target_pose,
                        post_error,
                        f"direct_metric_hard_force:{force_n:.2f}",
                        None,
                    )

            if post_error is None or self._active_error_units != "meters":
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=current_pose,
                    stiffness=self.stiffness,
                    damping=self.damping,
                )
                self.sleep_for(self.settle_s)
                self.get_logger().info(
                    "direct metric correction rejected no metric observation "
                    f"mapping={mapping_name} status={post_status}"
                )
                continue

            post_norm = float(np.linalg.norm(post_error))
            if post_norm < best_seen_norm:
                best_seen_norm = post_norm
                best_seen_name = mapping_name
            improvement_m = start_norm - post_norm
            self.get_logger().info(
                "direct metric correction result "
                f"mapping={mapping_name} before={start_norm:.4f}m "
                f"after={post_norm:.4f}m improvement={improvement_m:.4f}m"
            )
            if improvement_m >= self.direct_metric_min_improvement_m:
                self._direct_metric_mapping_index = mapping_index
                return (
                    target_pose,
                    post_error,
                    (
                        "direct_metric_improved:"
                        f"{mapping_name}:{start_norm:.4f}->{post_norm:.4f}m"
                    ),
                    correction_xy,
                )

            self.set_pose_target(
                move_robot=move_robot,
                pose=current_pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            self.sleep_for(self.settle_s)

        return (
            current_pose,
            error_m,
            (
                "direct_metric_no_improvement:"
                f"best={best_seen_name}:{start_norm:.4f}->{best_seen_norm:.4f}m"
            ),
            None,
        )

    def _anchor_handoff_metrics(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        start_index: int,
        label: str,
    ) -> dict[str, object]:
        metrics: dict[str, object] = {
            "anchor_label": label,
            "anchor_template_name": self.template_name,
            "anchor_template_path": str(self.template_path),
            "anchor_template_source": self.template_candidate.source,
            "anchor_waypoints": len(self.waypoints),
            "anchor_start_index": int(start_index),
            "anchor_force_n": None,
            "anchor_max_force_n": float(self.max_force_magnitude_n),
            "anchor_visual_status": "not_observed",
            "anchor_visual_error_norm_px": None,
            "anchor_visual_error_px": None,
            "anchor_visual_error_norm_m": None,
            "anchor_visual_error_m": None,
            "anchor_visual_pose_valid": False,
            "anchor_visual_pose_confidence": 0.0,
            "anchor_visual_pose_reprojection_error_px": None,
            "anchor_target_visible": False,
            "anchor_target_confidence": 0.0,
            "anchor_plug_visible": False,
            "anchor_plug_confidence": 0.0,
            "anchor_insertion_events": list(self.insertion_events),
        }
        observation = get_observation()
        if observation is None:
            metrics["anchor_visual_status"] = "no_observation"
            return metrics

        force_n = self._force_from_observation(observation)
        metrics["anchor_force_n"] = float(force_n)
        metrics["anchor_max_force_n"] = float(self.max_force_magnitude_n)
        cv_obs = self.cv_pipeline.process_frame(
            raw_img=observation.center_image,
            task_msg=task,
            timestamp=time.time(),
            camera="center",
        )
        error = np.array([cv_obs.visual_error.du, cv_obs.visual_error.dv], dtype=np.float64)
        error_norm = float(np.linalg.norm(error))
        pose_estimate = cv_obs.pose_estimate
        pose_error = None
        pose_error_norm = None
        if pose_estimate is not None and pose_estimate.valid:
            pose_error = np.array(pose_estimate.lateral_error_m, dtype=np.float64)
            pose_error_norm = float(np.linalg.norm(pose_error))
        metrics.update(
            {
                "anchor_visual_error_norm_px": error_norm,
                "anchor_visual_error_px": error.astype(float).tolist(),
                "anchor_visual_error_norm_m": pose_error_norm,
                "anchor_visual_error_m": (
                    pose_error.astype(float).tolist() if pose_error is not None else None
                ),
                "anchor_visual_pose_valid": bool(
                    pose_estimate is not None and pose_estimate.valid
                ),
                "anchor_visual_pose_confidence": float(
                    pose_estimate.confidence if pose_estimate is not None else 0.0
                ),
                "anchor_visual_pose_reprojection_error_px": (
                    float(pose_estimate.reprojection_error_px)
                    if pose_estimate is not None
                    else None
                ),
                "anchor_target_visible": bool(cv_obs.target.visible),
                "anchor_target_confidence": float(cv_obs.target.confidence),
                "anchor_plug_visible": bool(cv_obs.plug.visible),
                "anchor_plug_confidence": float(cv_obs.plug.confidence),
                "anchor_insertion_events": list(self.insertion_events),
            }
        )
        if not cv_obs.target.visible or cv_obs.target.confidence < self.min_target_conf:
            metrics["anchor_visual_status"] = "target_not_visible"
        elif not cv_obs.plug.visible or cv_obs.plug.confidence < self.min_plug_conf:
            metrics["anchor_visual_status"] = "plug_not_visible"
        else:
            metrics["anchor_visual_status"] = "ready_for_local_correction"
        self.get_logger().info(
            "SC replay anchor handoff: "
            f"template={self.template_name} start_index={start_index} "
            f"status={metrics['anchor_visual_status']} "
            f"err_px={metrics['anchor_visual_error_norm_px']} "
            f"err_m={metrics['anchor_visual_error_norm_m']} "
            f"target_conf={metrics['anchor_target_confidence']:.2f} "
            f"plug_conf={metrics['anchor_plug_confidence']:.2f} "
            f"force_n={force_n:.2f} events={self.insertion_events}"
        )
        return metrics

    def _observe_visual_error(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
    ) -> tuple[object | None, np.ndarray | None, str]:
        observation = get_observation()
        if observation is None:
            return None, None, "no_observation"
        cv_obs = self.cv_pipeline.process_frame(
            raw_img=observation.center_image,
            task_msg=task,
            timestamp=time.time(),
            camera="center",
        )
        if not cv_obs.target.visible or cv_obs.target.confidence < self.min_target_conf:
            return observation, None, "target_not_visible"
        if not cv_obs.plug.visible or cv_obs.plug.confidence < self.min_plug_conf:
            return observation, None, "plug_not_visible"
        pose_estimate = cv_obs.pose_estimate
        if (
            self.use_pose_estimate
            and pose_estimate is not None
            and pose_estimate.valid
            and pose_estimate.confidence >= self.pose_min_conf
            and pose_estimate.reprojection_error_px <= self.pose_max_reprojection_px
        ):
            error_m = np.array(pose_estimate.lateral_error_m, dtype=np.float64)
            self._active_error_units = "meters"
            return observation, error_m, (
                f"err={float(np.linalg.norm(error_m)):.4f}m "
                f"target_conf={cv_obs.target.confidence:.2f} "
                f"plug_conf={cv_obs.plug.confidence:.2f} "
                f"pose_conf={pose_estimate.confidence:.2f} "
                f"reproj={pose_estimate.reprojection_error_px:.1f}px"
            )
        error = np.array([cv_obs.visual_error.du, cv_obs.visual_error.dv], dtype=np.float64)
        self._active_error_units = "pixels"
        return observation, error, (
            f"err={float(np.linalg.norm(error)):.1f}px "
            f"target_conf={cv_obs.target.confidence:.2f} "
            f"plug_conf={cv_obs.plug.confidence:.2f}"
        )

    def _move_and_observe(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        pose: Pose,
        reason: str,
    ) -> tuple[object | None, np.ndarray | None, str]:
        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=self.stiffness,
            damping=self.damping,
        )
        send_feedback(reason)
        self.sleep_for(self.settle_s)
        return self._observe_visual_error(task=task, get_observation=get_observation)

    def _estimate_local_jacobian(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        base_pose: Pose,
        base_error: np.ndarray,
    ) -> np.ndarray | None:
        columns = []
        base_error_units = self._active_error_units
        for axis_index, axis_name in enumerate(("x", "y")):
            column = None
            last_status = "not_attempted"
            for sign in (1.0, -1.0):
                signed_step = sign * self.probe_step_m
                probe_delta = np.zeros(3, dtype=np.float64)
                probe_delta[axis_index] = signed_step
                probe_pose = self._offset_pose_xyz(base_pose, probe_delta)
                observation, probe_error, status = self._move_and_observe(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    pose=probe_pose,
                    reason=f"visual-servo probe {'+' if sign > 0 else '-'}{axis_name}",
                )
                last_status = status
                if observation is not None:
                    force_n = self._force_from_observation(observation)
                    if force_n >= self.hard_force_n:
                        self.get_logger().error(
                            "visual-servo probe hard force "
                            f"axis={axis_name} sign={sign:+.0f} force_n={force_n:.2f}"
                        )
                        self.set_pose_target(
                            move_robot=move_robot,
                            pose=base_pose,
                            stiffness=self.stiffness,
                            damping=self.damping,
                        )
                        self.sleep_for(self.settle_s)
                        return None
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=base_pose,
                    stiffness=self.stiffness,
                    damping=self.damping,
                )
                self.sleep_for(self.settle_s)
                if probe_error is None:
                    self.get_logger().warning(
                        "visual-servo probe failed "
                        f"axis={axis_name} sign={sign:+.0f}: {status}"
                    )
                    continue
                if self._active_error_units != base_error_units:
                    self.get_logger().warning(
                        "visual-servo probe skipped after error-unit switch "
                        f"axis={axis_name} sign={sign:+.0f} "
                        f"from={base_error_units} to={self._active_error_units}"
                    )
                    continue
                column = (probe_error - base_error) / max(abs(signed_step), 1e-9)
                if sign < 0.0:
                    column *= -1.0
                break
            if column is None:
                self.get_logger().warning(
                    f"visual-servo probe failed axis={axis_name}: {last_status}"
                )
                return None
            columns.append(column)

        jacobian = np.column_stack(columns)
        det = float(np.linalg.det(jacobian))
        if not np.isfinite(det) or abs(det) < 1e-3:
            self.get_logger().warning(
                f"visual-servo jacobian rejected det={det:.6f} J={jacobian.tolist()}"
            )
            return None
        self.get_logger().info(
            f"visual-servo jacobian det={det:.3f} J={jacobian.astype(float).tolist()}"
        )
        return jacobian

    def _coarse_visual_search(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        base_pose: Pose,
        base_error: np.ndarray,
    ) -> tuple[Pose, np.ndarray, str]:
        """Grid-search a safe lateral correction when the replay handoff is far off.

        The existing visual servo estimates a local Jacobian. That is useful only
        once the visual residual is already small. This coarse pass deliberately
        avoids a Jacobian and only asks whether nearby lateral offsets improve the
        plug-target pixel error before continuing with the local controller.
        """
        if self.coarse_search_radius_m <= 0.0:
            return base_pose, base_error, "coarse_disabled_zero_radius"

        offsets: list[tuple[float, float]] = [(0.0, 0.0)]
        steps = int(math.ceil(self.coarse_search_radius_m / self.coarse_search_step_m))
        for ix in range(-steps, steps + 1):
            for iy in range(-steps, steps + 1):
                dx = float(ix) * self.coarse_search_step_m
                dy = float(iy) * self.coarse_search_step_m
                if dx == 0.0 and dy == 0.0:
                    continue
                radius = math.hypot(dx, dy)
                if radius <= self.coarse_search_radius_m + 1e-9:
                    offsets.append((dx, dy))
        offsets = sorted(offsets, key=lambda pair: (math.hypot(pair[0], pair[1]), pair[0], pair[1]))
        offsets = offsets[: self.coarse_search_max_candidates]

        best_pose = self._copy_pose(base_pose)
        best_error = np.array(base_error, dtype=np.float64)
        best_norm = float(np.linalg.norm(best_error))
        start_norm = best_norm
        error_units = self._active_error_units
        min_improvement = self._coarse_min_improvement(error_units)

        for index, (dx, dy) in enumerate(offsets[1:], start=1):
            candidate_pose = self._offset_pose_xyz(
                base_pose, np.array([dx, dy, 0.0], dtype=np.float64)
            )
            observation, error, status = self._move_and_observe(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                pose=candidate_pose,
                reason=f"visual coarse search {index}/{len(offsets) - 1}",
            )
            if observation is not None:
                force_n = self._force_from_observation(observation)
                if force_n >= self.hard_force_n:
                    self.get_logger().error(
                        f"visual coarse search hard force index={index} force_n={force_n:.2f}"
                    )
                    break
            if error is None:
                self.get_logger().info(
                    f"visual coarse search skipped index={index} offset=({dx:.3f},{dy:.3f}) "
                    f"status={status}"
                )
                continue
            if self._active_error_units != error_units:
                self.get_logger().info(
                    "visual coarse search skipped candidate due to error-unit switch "
                    f"from {error_units} to {self._active_error_units}"
                )
                continue
            norm = float(np.linalg.norm(error))
            self.get_logger().info(
                f"visual coarse search index={index} offset=({dx:.3f},{dy:.3f}) "
                f"err_norm={self._format_error_norm(norm, error_units)} "
                f"best={self._format_error_norm(best_norm, error_units)}"
            )
            if norm < best_norm:
                best_norm = norm
                best_error = np.array(error, dtype=np.float64)
                best_pose = candidate_pose

        if best_norm < start_norm - min_improvement:
            self.set_pose_target(
                move_robot=move_robot,
                pose=best_pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            self.sleep_for(self.settle_s)
            return (
                best_pose,
                best_error,
                "coarse_improved:"
                f"{self._format_error_norm(start_norm, error_units)}->"
                f"{self._format_error_norm(best_norm, error_units)}",
            )

        self.set_pose_target(
            move_robot=move_robot,
            pose=base_pose,
            stiffness=self.stiffness,
            damping=self.damping,
        )
        self.sleep_for(self.settle_s)
        return (
            base_pose,
            base_error,
            "coarse_no_improvement:"
            f"{self._format_error_norm(start_norm, error_units)}->"
            f"{self._format_error_norm(best_norm, error_units)}",
        )

    def _visual_center(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> tuple[Pose | None, str]:
        jacobian: np.ndarray | None = None
        total_correction = np.zeros(2, dtype=np.float64)
        dwell = 0
        last_pose: Pose | None = None
        saw_valid_error = False

        for step in range(self.max_center_steps):
            observation, error, status = self._observe_visual_error(
                task=task,
                get_observation=get_observation,
            )
            if observation is None:
                self.sleep_for(self.settle_s)
                continue
            force_n = self._force_from_observation(observation)
            if force_n >= self.hard_force_n:
                return last_pose, f"visual_center_hard_force:{force_n:.2f}"
            current_pose = self._pose_from_observation(observation)
            last_pose = current_pose
            if error is None:
                send_feedback(f"visual center waiting: {status}")
                self.get_logger().info(f"visual center waiting step={step}: {status}")
                self.sleep_for(self.settle_s)
                continue

            error_norm = float(np.linalg.norm(error))
            error_units = self._active_error_units
            tolerance = self._error_tolerance(error_units)
            max_initial_error = self._max_initial_error(error_units)
            coarse_accept_max_error = self._coarse_accept_max_error(error_units)
            max_correction_total = self._max_correction_total(error_units)
            if not saw_valid_error:
                saw_valid_error = True
                if error_norm > max_initial_error:
                    if not self.coarse_search_enabled:
                        return current_pose, (
                            "visual_center_initial_error_too_high:"
                            f"{self._format_error_norm(error_norm, error_units)}"
                        )
                    current_pose, error, coarse_reason = self._coarse_visual_search(
                        task=task,
                        get_observation=get_observation,
                        move_robot=move_robot,
                        send_feedback=send_feedback,
                        base_pose=current_pose,
                        base_error=error,
                    )
                    last_pose = current_pose
                    error_norm = float(np.linalg.norm(error))
                    error_units = self._active_error_units
                    tolerance = self._error_tolerance(error_units)
                    coarse_accept_max_error = self._coarse_accept_max_error(error_units)
                    max_correction_total = self._max_correction_total(error_units)
                    self.get_logger().info(
                        f"visual coarse search result before centering: {coarse_reason}"
                    )
                    if error_norm > coarse_accept_max_error:
                        return current_pose, (
                            "visual_center_initial_error_too_high_after_coarse:"
                            f"{self._format_error_norm(error_norm, error_units)}:"
                            f"{coarse_reason}"
                        )
            if step == 0 and error_norm > max_initial_error:
                if not self.coarse_search_enabled:
                    return current_pose, (
                        "visual_center_initial_error_too_high:"
                        f"{self._format_error_norm(error_norm, error_units)}"
                    )
                current_pose, error, coarse_reason = self._coarse_visual_search(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    base_pose=current_pose,
                    base_error=error,
                )
                last_pose = current_pose
                error_norm = float(np.linalg.norm(error))
                error_units = self._active_error_units
                tolerance = self._error_tolerance(error_units)
                coarse_accept_max_error = self._coarse_accept_max_error(error_units)
                max_correction_total = self._max_correction_total(error_units)
                self.get_logger().info(
                    f"visual coarse search result before centering: {coarse_reason}"
                )
                if error_norm > coarse_accept_max_error:
                    return current_pose, (
                        "visual_center_initial_error_too_high_after_coarse:"
                        f"{self._format_error_norm(error_norm, error_units)}:"
                        f"{coarse_reason}"
                    )
            self.get_logger().info(
                f"visual center step={step} error_units={error_units} "
                f"error={error.astype(float).tolist()} "
                f"norm={self._format_error_norm(error_norm, error_units)} "
                f"force={force_n:.2f}"
            )
            if error_norm <= tolerance:
                dwell += 1
                if dwell >= self.center_dwell_steps:
                    return current_pose, "visual_centered"
                self.sleep_for(self.settle_s)
                continue
            dwell = 0

            if error_units == "meters" and self.direct_metric_correction_enabled:
                (
                    direct_pose,
                    direct_error,
                    direct_reason,
                    accepted_delta,
                ) = self._apply_direct_metric_correction(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    current_pose=current_pose,
                    error_m=error,
                    total_correction=total_correction,
                    max_correction_total=max_correction_total,
                    step=step,
                )
                if accepted_delta is None:
                    return direct_pose, direct_reason
                total_correction += accepted_delta
                last_pose = direct_pose
                jacobian = None
                if direct_error is not None:
                    self.get_logger().info(
                        "direct metric correction accepted "
                        f"reason={direct_reason} "
                        f"total_correction={total_correction.astype(float).tolist()}"
                    )
                continue

            if jacobian is None or (
                self.jacobian_refresh_steps > 0 and step % self.jacobian_refresh_steps == 0
            ):
                jacobian = self._estimate_local_jacobian(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    base_pose=current_pose,
                    base_error=error,
                )
                if jacobian is None:
                    return current_pose, "visual_center_no_jacobian"

            correction_xy = -np.linalg.pinv(jacobian) @ error
            correction_norm = float(np.linalg.norm(correction_xy))
            if correction_norm > self.max_correction_step_m and correction_norm > 1e-9:
                correction_xy *= self.max_correction_step_m / correction_norm
            if float(np.linalg.norm(total_correction + correction_xy)) > max_correction_total:
                return current_pose, "visual_center_correction_budget_exhausted"
            total_correction += correction_xy

            delta = np.array([correction_xy[0], correction_xy[1], 0.0], dtype=np.float64)
            target_pose = self._offset_pose_xyz(current_pose, delta)
            _, post_error, post_status = self._move_and_observe(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                pose=target_pose,
                reason=f"visual center correction step={step}",
            )
            if post_error is not None:
                if self._active_error_units != error_units:
                    jacobian = None
                    continue
                post_norm = float(np.linalg.norm(post_error))
                worsen_margin = 0.003 if error_units == "meters" else 12.0
                if post_norm > error_norm + worsen_margin:
                    self.get_logger().warning(
                        "visual center correction worsened; refreshing jacobian "
                        f"before={self._format_error_norm(error_norm, error_units)} "
                        f"after={self._format_error_norm(post_norm, error_units)}"
                    )
                    jacobian = None
            else:
                self.get_logger().warning(
                    f"visual center post-correction observation unavailable: {post_status}"
                )

        return last_pose, "visual_center_timeout"

    def _template_insert_axis(self, start_index: int) -> np.ndarray:
        if self.insert_axis_mode in {"terminal", "tail", "final"}:
            start_index = max(0, len(self.waypoints) - 1 - self.terminal_axis_waypoints)
        start_index = int(np.clip(start_index, 0, len(self.waypoints) - 1))
        start = self._pose_position_np(self.waypoints[start_index].pose)
        end = self._pose_position_np(self.waypoints[-1].pose)
        axis = end - start
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        axis = axis / norm
        self.get_logger().info(
            "SC terminal insert axis "
            f"mode={self.insert_axis_mode} start_index={start_index} "
            f"end_index={len(self.waypoints) - 1} "
            f"axis={axis.astype(float).tolist()} scale={self.insert_axis_scale:.3f}"
        )
        return axis * self.insert_axis_scale

    def _pre_commit_gate(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        pose: Pose,
    ) -> tuple[bool, Pose, str]:
        if self.terminal_pre_commit_hold_s <= 0.0:
            return True, pose, "pre_commit_hold_disabled"

        self.set_pose_target(
            move_robot=move_robot,
            pose=pose,
            stiffness=self.stiffness,
            damping=self.damping,
        )
        start_time = time.time()
        first_force_n: float | None = None
        latest_pose = self._copy_pose(pose)
        latest_visual_status = "not_checked"
        latest_visual_norm: float | None = None
        latest_visual_units = "unknown"

        while time.time() - start_time < self.terminal_pre_commit_hold_s:
            observation, error, status = self._observe_visual_error(
                task=task,
                get_observation=get_observation,
            )
            latest_visual_status = status
            if observation is None:
                self.sleep_for(min(self.settle_s, 0.10))
                continue

            latest_pose = self._pose_from_observation(observation)
            force_n = self._force_from_observation(observation)
            if first_force_n is None:
                first_force_n = force_n
            if force_n >= self.terminal_pre_commit_max_force_n:
                reason = f"pre_commit_force:{force_n:.2f}"
                send_feedback(reason)
                return False, latest_pose, reason
            if force_n - first_force_n >= self.terminal_pre_commit_force_rise_n:
                reason = f"pre_commit_force_rise:{first_force_n:.2f}->{force_n:.2f}"
                send_feedback(reason)
                return False, latest_pose, reason

            if error is None:
                if self.terminal_pre_commit_require_visual:
                    reason = f"pre_commit_visual_unavailable:{status}"
                    send_feedback(reason)
                    return False, latest_pose, reason
                self.sleep_for(min(self.settle_s, 0.10))
                continue

            latest_visual_units = self._active_error_units
            latest_visual_norm = float(np.linalg.norm(error))
            visual_limit = (
                self._error_tolerance(latest_visual_units)
                * self.terminal_pre_commit_visual_margin
            )
            if latest_visual_norm > visual_limit:
                reason = (
                    "pre_commit_visual_drift:"
                    f"{self._format_error_norm(latest_visual_norm, latest_visual_units)}"
                )
                send_feedback(reason)
                return False, latest_pose, reason
            self.sleep_for(min(self.settle_s, 0.10))

        reason = (
            "pre_commit_stable:"
            f"force={self.max_force_magnitude_n:.2f}:"
            f"visual={latest_visual_status}:"
            f"norm={latest_visual_norm}:units={latest_visual_units}"
        )
        send_feedback(reason)
        return True, latest_pose, reason

    def _feather_insert(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        start_pose: Pose,
        insert_axis: np.ndarray,
    ) -> tuple[str, Pose]:
        pose = self._copy_pose(start_pose)
        gate_ok, pose, gate_reason = self._pre_commit_gate(
            task=task,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
            pose=pose,
        )
        if not gate_ok:
            self.get_logger().warning(f"SC pre-commit gate rejected: {gate_reason}")
            return gate_reason, pose

        self.guarded_recovery.reset()
        self.guarded_recovery.begin_insert_attempt(
            current_pose=pose,
            now_s=time.time(),
        )
        last_pose_for_speed = self._pose_position_np(pose)
        last_time_for_speed = time.time()
        retry_count_seen = 0
        for step in range(self.max_insert_steps):
            if self.stop_on_insertion_event and self.insertion_events:
                return "insertion_event", pose
            force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)
            observation = get_observation()
            if observation is not None:
                force_n = self._force_from_observation(observation)
                if not self.use_guarded_recovery and force_n >= self.hard_force_n:
                    return f"insert_hard_force:{force_n:.2f}", pose
                pose = self._pose_from_observation(observation)
            now_s = time.time()
            current_pos = self._pose_position_np(pose)
            dt = max(now_s - last_time_for_speed, 1.0e-6)
            tcp_linear_speed_mps = float(np.linalg.norm(current_pos - last_pose_for_speed) / dt)
            raw_target_pose = self._offset_pose_xyz(pose, insert_axis * self.insert_step_m)
            if self.use_guarded_recovery:
                decision = self.guarded_recovery.evaluate_insert_step(
                    current_pose=pose,
                    proposed_pose=raw_target_pose,
                    insert_direction_world=insert_axis,
                    force_n=force_n,
                    tcp_linear_speed_mps=tcp_linear_speed_mps,
                    now_s=now_s,
                )
                target_pose = decision.bounded_pose
                if decision.should_retry and decision.retry_target_pose is not None:
                    retry_count_seen += 1
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=decision.retry_target_pose,
                        stiffness=self.stiffness,
                        damping=self.damping,
                    )
                    send_feedback(
                        "visual-servo guarded reseat/backout "
                        f"step={step} reason={decision.trigger_reason} "
                        f"retry={decision.retries_used}"
                    )
                    self.sleep_for(max(self.settle_s, self.period_s))
                    pose = decision.retry_target_pose
                    self.guarded_recovery.begin_insert_attempt(
                        current_pose=pose,
                        now_s=time.time(),
                    )
                    last_pose_for_speed = self._pose_position_np(pose)
                    last_time_for_speed = time.time()
                    continue
                if decision.should_fail:
                    reason = decision.termination_reason or "guarded_insert_failed"
                    return reason, pose
            else:
                decision = None
                target_pose = raw_target_pose
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            pose = target_pose
            send_feedback(
                "visual-servo feather insert "
                f"step={step} profile={self.final_centimeter_profile} "
                f"step_m={self.insert_step_m:.5f} force_n={force_n:.2f} "
                f"guard={decision.trigger_reason if decision else None} "
                f"would_trigger={decision.would_trigger if decision else False} "
                f"retries={retry_count_seen}"
            )
            self.get_logger().info(
                "SC feather insert step "
                f"step={step} force_n={force_n:.2f} "
                f"target=({pose.position.x:.5f},{pose.position.y:.5f},{pose.position.z:.5f}) "
                f"guarded={self.use_guarded_recovery} "
                f"reason={decision.would_trigger_reason if decision else None} "
                f"retries={retry_count_seen}"
            )
            self.sleep_for(self.period_s)
            last_pose_for_speed = current_pos
            last_time_for_speed = now_s
        return "insert_step_limit", pose

    def _run_replay_remainder(
        self,
        *,
        start_index: int,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        offset_xyz: np.ndarray | None = None,
    ) -> Pose:
        if offset_xyz is None or not self.carry_replay_offset:
            offset_xyz = np.zeros(3, dtype=np.float64)
        last_pose = self._offset_pose_xyz(self.waypoints[start_index].pose, offset_xyz)
        hard_force_steps = 0
        for step, waypoint in enumerate(self.waypoints[start_index + 1 :], start=start_index + 1):
            loop_start = time.time()
            observation = get_observation()
            if observation is not None:
                force_n = self._force_from_observation(observation)
            else:
                force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)
            if force_n >= self.hard_force_n:
                hard_force_steps += 1
            else:
                hard_force_steps = 0
            if hard_force_steps >= self.hard_force_consecutive_steps:
                self.get_logger().error(
                    f"visual-servo fallback replay hard-force abort step={step} force_n={force_n:.2f}"
                )
                break
            target_pose = self._offset_pose_xyz(waypoint.pose, offset_xyz)
            self.set_pose_target(
                move_robot=move_robot,
                pose=target_pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            last_pose = target_pose
            if step % 20 == 0:
                self.get_logger().info(
                    f"visual-servo fallback replay step={step}/{len(self.waypoints)} force_n={force_n:.2f}"
                )
            send_feedback(f"visual-servo fallback replay step={step}")
            if self.stop_on_insertion_event and self.insertion_events:
                break
            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.period_s - elapsed))
        return last_pose

    def _build_visual_primitive_runtime(self) -> PrimitiveRuntime:
        primitives = {
            "replay_approach": CallbackPrimitive(
                "replay_approach",
                self._primitive_replay_approach,
                phase="REPLAY_APPROACH",
                description="Replay the fixed SC approach template to the visual-servo start.",
            ),
            "visual_servo": CallbackPrimitive(
                "visual_servo",
                self._primitive_visual_servo,
                phase="PORT_CENTERING",
                description="Deployable image-based centering using local Jacobian probing.",
            ),
            "guarded_insert": CallbackPrimitive(
                "guarded_insert",
                self._primitive_visual_guarded_insert,
                phase="FEATHERED_INSERT",
                description="Feathered insertion from a visually centered pose.",
            ),
            "replay_remainder": CallbackPrimitive(
                "replay_remainder",
                self._primitive_replay_remainder,
                phase="REPLAY_REMAINDER",
                description="Fallback to the remaining replay trajectory if visual centering fails.",
            ),
            "final_hold": CallbackPrimitive(
                "final_hold",
                self._primitive_final_hold,
                phase="FINAL_HOLD",
                description="Hold the terminal pose before returning wrapper status.",
            ),
        }
        transitions = {
            ("replay_approach", "ok"): "visual_servo",
            ("visual_servo", "centered"): "guarded_insert",
            ("visual_servo", "fallback"): "replay_remainder",
            ("guarded_insert", "done"): "final_hold",
            ("replay_remainder", "done"): "final_hold",
        }
        return PrimitiveRuntime(
            primitives,
            transitions,
            logger=self.get_logger(),
        )

    def _primitive_replay_approach(self, context: PrimitiveContext) -> PrimitiveResult:
        start_index = int(context.state["start_index"])
        hard_force_steps = 0
        for step, waypoint in enumerate(self.waypoints[: start_index + 1]):
            loop_start = time.time()
            observation = context.get_observation()
            if observation is not None:
                force_n = self._force_from_observation(observation)
            else:
                force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)

            hard_force_steps = hard_force_steps + 1 if force_n >= self.hard_force_n else 0
            if hard_force_steps >= self.hard_force_consecutive_steps:
                reason = f"visual_servo_replay_hard_force:{force_n:.2f}"
                self.get_logger().error(
                    f"visual-servo replay abort during approach step={step} force_n={force_n:.2f}"
                )
                return PrimitiveResult(
                    name="replay_approach",
                    status="abort",
                    reason=reason,
                    terminal_success=False,
                    retryable=False,
                )

            self.set_pose_target(
                move_robot=context.move_robot,
                pose=waypoint.pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            context.state["last_pose"] = waypoint.pose
            if step % 20 == 0:
                self.get_logger().info(
                    f"visual-servo replay approach step={step}/{start_index} force_n={force_n:.2f}"
                )
            context.send_feedback(f"visual-servo replay approach step={step}")
            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.period_s - elapsed))
        anchor_metrics = self._anchor_handoff_metrics(
            task=context.task,
            get_observation=context.get_observation,
            start_index=start_index,
            label="post_replay_approach",
        )
        context.state["anchor_handoff_metrics"] = anchor_metrics
        return PrimitiveResult(
            name="replay_approach",
            status="ok",
            reason="replay_start_reached",
            metadata=anchor_metrics,
        )

    def _primitive_visual_servo(self, context: PrimitiveContext) -> PrimitiveResult:
        centered_pose, center_reason = self._visual_center(
            task=context.task,
            get_observation=context.get_observation,
            move_robot=context.move_robot,
            send_feedback=context.send_feedback,
        )
        self.get_logger().info(f"visual-servo center result: {center_reason}")
        if centered_pose is None:
            centered_pose = self.waypoints[int(context.state["start_index"])].pose
        context.state["centered_pose"] = centered_pose
        context.state["center_reason"] = center_reason
        start_pose = self.waypoints[int(context.state["start_index"])].pose
        context.state["replay_offset_xyz"] = (
            self._pose_position_np(centered_pose) - self._pose_position_np(start_pose)
        )
        if center_reason == "visual_centered":
            return PrimitiveResult(
                name="visual_servo",
                status="centered",
                reason=center_reason,
            )
        return PrimitiveResult(
            name="visual_servo",
            status="fallback",
            reason=center_reason,
        )

    def _primitive_visual_guarded_insert(
        self, context: PrimitiveContext
    ) -> PrimitiveResult:
        start_index = int(context.state["start_index"])
        centered_pose = context.state.get("centered_pose") or self.waypoints[start_index].pose
        insert_axis = self._template_insert_axis(start_index)
        insert_reason, final_pose = self._feather_insert(
            task=context.task,
            get_observation=context.get_observation,
            move_robot=context.move_robot,
            send_feedback=context.send_feedback,
            start_pose=centered_pose,
            insert_axis=insert_axis,
        )
        context.state["final_pose"] = final_pose
        context.state["insert_reason"] = insert_reason
        return PrimitiveResult(
            name="guarded_insert",
            status="done",
            reason=insert_reason,
            metadata={"insertion_events": list(self.insertion_events)},
        )

    def _primitive_replay_remainder(self, context: PrimitiveContext) -> PrimitiveResult:
        start_index = int(context.state["start_index"])
        final_pose = self._run_replay_remainder(
            start_index=start_index,
            get_observation=context.get_observation,
            move_robot=context.move_robot,
            send_feedback=context.send_feedback,
            offset_xyz=context.state.get("replay_offset_xyz"),
        )
        center_reason = str(context.state.get("center_reason") or "visual_center_unknown")
        insert_reason = f"fallback_replay_after:{center_reason}"
        if self.fallback_guarded_insert_enabled and not self.insertion_events:
            fallback_reason, final_pose = self._feather_insert(
                task=context.task,
                get_observation=context.get_observation,
                move_robot=context.move_robot,
                send_feedback=context.send_feedback,
                start_pose=final_pose,
                insert_axis=self._template_insert_axis(start_index),
            )
            insert_reason = f"{insert_reason}+guarded_insert:{fallback_reason}"
        context.state["final_pose"] = final_pose
        context.state["insert_reason"] = insert_reason
        return PrimitiveResult(
            name="replay_remainder",
            status="done",
            reason=insert_reason,
        )

    def _primitive_final_hold(self, context: PrimitiveContext) -> PrimitiveResult:
        final_pose = context.state.get("final_pose") or context.state.get("centered_pose")
        if final_pose is None:
            final_pose = self.waypoints[int(context.state["start_index"])].pose
        if self.final_hold_s > 0.0:
            self._hold_pose(
                move_robot=context.move_robot,
                pose=final_pose,
                hold_s=self.final_hold_s,
                send_feedback=context.send_feedback,
                reason="visual-servo final hold",
            )
        return PrimitiveResult(
            name="final_hold",
            status="done",
            reason=str(context.state.get("insert_reason") or "final_hold_complete"),
            terminal_success=True,
            retryable=False,
            metadata={"insertion_events": list(self.insertion_events)},
        )

    def _insert_cable_primitive_runtime(
        self,
        *,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        start_index: int,
        start_time: float,
    ) -> bool:
        context = PrimitiveContext(
            task=task,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
            branch="sc",
            template=f"sc_replay_visual_servo:{self.template_name}",
            started_s=start_time,
            metadata={
                "template_name": self.template_name,
                "template_path": str(self.template_path),
                "template_source": self.template_candidate.source,
                "template_metadata": dict(self.template_metadata),
            },
            state={
                "start_index": start_index,
                "template_name": self.template_name,
                "template_path": str(self.template_path),
            },
        )
        result, history = self._build_visual_primitive_runtime().run(
            start_primitive="replay_approach",
            context=context,
        )
        self.primitive_runtime_history = history
        self.get_logger().info(
            "RunSCVisualServoReplay primitive runtime exiting: "
            f"status={result.status} reason={result.reason} "
            f"template={self.template_name} "
            f"events={self.insertion_events} "
            f"max_force_n={self.max_force_magnitude_n:.2f} "
            f"duration_s={time.time() - start_time:.2f} "
            f"history={history}"
        )
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"RunSCVisualServoReplay.insert_cable() task: {self._task_summary(task)}"
        )
        self._select_replay_template_for_task(task)
        self.cv_pipeline.reset()
        hard_force_steps = 0
        start_time = time.time()

        if self.visual_start_fraction >= 0.0:
            start_index = int(
                np.clip(
                    round(self.visual_start_fraction * max(0, len(self.waypoints) - 1)),
                    0,
                    len(self.waypoints) - 1,
                )
            )
        else:
            start_index = int(np.clip(self.visual_start_waypoint, 0, len(self.waypoints) - 1))

        if self.primitive_runtime_enabled:
            return self._insert_cable_primitive_runtime(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                start_index=start_index,
                start_time=start_time,
            )

        for step, waypoint in enumerate(self.waypoints[: start_index + 1]):
            loop_start = time.time()
            observation = get_observation()
            if observation is not None:
                force_n = self._force_from_observation(observation)
            else:
                force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)

            if force_n >= self.hard_force_n:
                hard_force_steps += 1
            else:
                hard_force_steps = 0
            if hard_force_steps >= self.hard_force_consecutive_steps:
                self.get_logger().error(
                    f"visual-servo replay abort during approach step={step} force_n={force_n:.2f}"
                )
                return True

            self.set_pose_target(
                move_robot=move_robot,
                pose=waypoint.pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            if step % 20 == 0:
                self.get_logger().info(
                    f"visual-servo replay approach step={step}/{start_index} force_n={force_n:.2f}"
                )
            send_feedback(f"visual-servo replay approach step={step}")
            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.period_s - elapsed))

        self._anchor_handoff_metrics(
            task=task,
            get_observation=get_observation,
            start_index=start_index,
            label="post_replay_approach_nonprimitive",
        )

        centered_pose, center_reason = self._visual_center(
            task=task,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
        )
        self.get_logger().info(f"visual-servo center result: {center_reason}")
        if centered_pose is None:
            centered_pose = self.waypoints[start_index].pose

        if center_reason == "visual_centered":
            insert_axis = self._template_insert_axis(start_index)
            insert_reason, centered_pose = self._feather_insert(
                task=task,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                start_pose=centered_pose,
                insert_axis=insert_axis,
            )
        else:
            centered_pose = self._run_replay_remainder(
                start_index=start_index,
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                offset_xyz=(
                    self._pose_position_np(centered_pose)
                    - self._pose_position_np(self.waypoints[start_index].pose)
                ),
            )
            insert_reason = f"fallback_replay_after:{center_reason}"
            if self.fallback_guarded_insert_enabled and not self.insertion_events:
                fallback_reason, centered_pose = self._feather_insert(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    start_pose=centered_pose,
                    insert_axis=self._template_insert_axis(start_index),
                )
                insert_reason = f"{insert_reason}+guarded_insert:{fallback_reason}"

        if self.final_hold_s > 0.0:
            self._hold_pose(
                move_robot=move_robot,
                pose=centered_pose,
                hold_s=self.final_hold_s,
                send_feedback=send_feedback,
                reason="visual-servo final hold",
            )

        self.get_logger().info(
            "RunSCVisualServoReplay exiting: "
            f"template={self.template_name} center={center_reason} insert={insert_reason} "
            f"events={self.insertion_events} "
            f"max_force_n={self.max_force_magnitude_n:.2f} "
            f"duration_s={time.time() - start_time:.2f}"
        )
        return True
