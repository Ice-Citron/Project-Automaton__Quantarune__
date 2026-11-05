from __future__ import annotations

import math
import os
import time

import numpy as np

from aic_task_interfaces.msg import Task

from .RunACTCompetitor import _parse_bool, _parse_float
from .RunACTStaged import RunACTStaged
from .vision_target_features import CenterPortCVPipeline, CenterPortFeatureExtractor


class RunACTStagedVisionGate(RunACTStaged):
    """ACT48 staged wrapper with a broader vision supervision layer.

    This keeps the ACT48 checkpoint untouched and reuses Agent E's center-camera
    feature extractor outside the policy to supervise staged execution:
    - require a stable visual lock before graduating from approach to align
    - continuously recenter the staged pose target using image bearing
    - use confidence/scale/temporal consistency before guarded insertion
    - fall back to a small local search when the target is weak or lost
    - bias retries laterally so they do not repeat the same bad line
    """

    _VALID_VISION_MODES = {"shadow", "advisory", "control"}

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.vision_mode = (
            os.environ.get("AIC_ACT_STAGED_VISION_MODE", "shadow").strip().lower()
        )
        if self.vision_mode not in self._VALID_VISION_MODES:
            raise ValueError(
                "AIC_ACT_STAGED_VISION_MODE must be one of "
                f"{sorted(self._VALID_VISION_MODES)}, got {self.vision_mode!r}"
            )
        self.vision_min_area_frac = _parse_float(
            "AIC_ACT_STAGED_VISION_MIN_AREA_FRAC", 2.0e-4
        )
        self.vision_max_area_frac = _parse_float(
            "AIC_ACT_STAGED_VISION_MAX_AREA_FRAC", 1.5e-1
        )
        self.vision_valid_threshold = _parse_float(
            "AIC_ACT_STAGED_VISION_VALID_THRESHOLD", 0.35
        )
        self.vision_min_segmentation_conf = _parse_float(
            "AIC_ACT_STAGED_VISION_MIN_SEGMENTATION_CONF", 0.42
        )
        self.vision_min_selection_conf = _parse_float(
            "AIC_ACT_STAGED_VISION_MIN_SELECTION_CONF", 0.55
        )
        self.vision_min_scale_norm = _parse_float(
            "AIC_ACT_STAGED_VISION_MIN_SCALE_NORM", 0.08
        )
        self.vision_align_bearing_tol_x = _parse_float(
            "AIC_ACT_STAGED_VISION_ALIGN_BEARING_TOL_X", 0.16
        )
        self.vision_align_bearing_tol_y = _parse_float(
            "AIC_ACT_STAGED_VISION_ALIGN_BEARING_TOL_Y", 0.18
        )
        self.vision_bias_gain_x_m = _parse_float(
            "AIC_ACT_STAGED_VISION_BIAS_GAIN_X_M", 0.0045
        )
        self.vision_bias_gain_y_m = _parse_float(
            "AIC_ACT_STAGED_VISION_BIAS_GAIN_Y_M", 0.0045
        )
        self.vision_bias_limit_m = _parse_float(
            "AIC_ACT_STAGED_VISION_BIAS_LIMIT_M", 0.0060
        )
        self.vision_search_radius_m = _parse_float(
            "AIC_ACT_STAGED_VISION_SEARCH_RADIUS_M", 0.0035
        )
        self.vision_search_max_radius_m = _parse_float(
            "AIC_ACT_STAGED_VISION_SEARCH_MAX_RADIUS_M", 0.0075
        )
        self.vision_search_period_s = _parse_float(
            "AIC_ACT_STAGED_VISION_SEARCH_PERIOD_S", 2.2
        )
        self.vision_bearing_ema_alpha = _parse_float(
            "AIC_ACT_STAGED_VISION_BEARING_EMA_ALPHA", 0.35
        )
        self.vision_align_lock_frames = max(
            1, int(round(_parse_float("AIC_ACT_STAGED_VISION_ALIGN_LOCK_FRAMES", 3.0)))
        )
        self.vision_insert_centered_frames = max(
            1,
            int(
                round(
                    _parse_float(
                        "AIC_ACT_STAGED_VISION_INSERT_CENTERED_FRAMES", 4.0
                    )
                )
            ),
        )
        self.vision_loss_tolerance_frames = max(
            0,
            int(
                round(
                    _parse_float(
                        "AIC_ACT_STAGED_VISION_LOSS_TOLERANCE_FRAMES", 2.0
                    )
                )
            ),
        )
        self.vision_retry_bias_gain_x_m = _parse_float(
            "AIC_ACT_STAGED_VISION_RETRY_BIAS_GAIN_X_M", 0.0060
        )
        self.vision_retry_bias_gain_y_m = _parse_float(
            "AIC_ACT_STAGED_VISION_RETRY_BIAS_GAIN_Y_M", 0.0060
        )
        self.vision_x_sign = _parse_float("AIC_ACT_STAGED_VISION_X_SIGN", 1.0)
        self.vision_y_sign = _parse_float("AIC_ACT_STAGED_VISION_Y_SIGN", 1.0)
        self.vision_require_valid_for_insert = _parse_bool(
            "AIC_ACT_STAGED_VISION_REQUIRE_VALID_FOR_INSERT", True
        )
        self.vision_enable_search = _parse_bool(
            "AIC_ACT_STAGED_VISION_ENABLE_SEARCH", True
        )

        self._vision_pipeline = CenterPortCVPipeline(
            extractor=CenterPortFeatureExtractor(
                min_area_frac=self.vision_min_area_frac,
                max_area_frac=self.vision_max_area_frac,
                valid_threshold=self.vision_valid_threshold,
            ),
            camera_name="wrist",
        )
        self._reset_vision_state()

        self.get_logger().info(
            "RunACTStagedVisionGate active with "
            f"mode={self.vision_mode} "
            f"seg_conf>={self.vision_min_segmentation_conf:.2f} "
            f"sel_conf>={self.vision_min_selection_conf:.2f} "
            f"scale>={self.vision_min_scale_norm:.2f} "
            f"bearing_tol=({self.vision_align_bearing_tol_x:.2f}, "
            f"{self.vision_align_bearing_tol_y:.2f}) "
            f"align_lock_frames={self.vision_align_lock_frames} "
            f"insert_centered_frames={self.vision_insert_centered_frames} "
            f"loss_tolerance_frames={self.vision_loss_tolerance_frames} "
            f"bias_gain=({self.vision_bias_gain_x_m:.4f}, "
            f"{self.vision_bias_gain_y_m:.4f}) "
            f"retry_bias=({self.vision_retry_bias_gain_x_m:.4f}, "
            f"{self.vision_retry_bias_gain_y_m:.4f}) "
            f"search_radius={self.vision_search_radius_m:.4f}->{self.vision_search_max_radius_m:.4f} "
            f"require_valid_for_insert={self.vision_require_valid_for_insert}"
        )

    def _reset_vision_state(self) -> None:
        self._vision_bearing_ema = np.zeros(2, dtype=np.float64)
        self._vision_ema_initialized = False
        self._vision_lock_frames = 0
        self._vision_centered_frames = 0
        self._vision_lost_frames = 0
        self._vision_last_locked_bearing = np.zeros(2, dtype=np.float64)
        self._vision_last_locked_scale = 0.0
        self._vision_frame_id = 0
        self._latest_cv_observation = None
        self._latest_structured_cv_observation = None
        self._last_guidance_context = None
        self._vision_pipeline.reset()

    def insert_cable(self, *args, **kwargs):
        self._reset_vision_state()
        return super().insert_cable(*args, **kwargs)

    def get_latest_cv_observation(self):
        return self._latest_structured_cv_observation

    def get_latest_cv_output_row(self):
        if self._latest_structured_cv_observation is None:
            return None
        return self._latest_structured_cv_observation.to_harness_row()

    def _cv_timestamp_s(self) -> float:
        try:
            return float(self.time_now().nanoseconds) * 1.0e-9
        except Exception:
            return time.time()

    def _build_cv_observation(self, obs_msg, task_msg):
        cv_observation = self._vision_pipeline.process_frame(
            raw_img=obs_msg.center_image,
            task_info=task_msg,
            timestamp=self._cv_timestamp_s(),
            frame_id=self._vision_frame_id,
            camera="wrist",
        )
        self._vision_frame_id += 1
        self._latest_structured_cv_observation = cv_observation
        return cv_observation.to_harness_row()

    @staticmethod
    def _rotate_local_offset(quaternion, local_offset: np.ndarray) -> np.ndarray:
        q_vec = np.array(
            [quaternion.x, quaternion.y, quaternion.z], dtype=np.float64
        )
        q_w = float(quaternion.w)
        t = 2.0 * np.cross(q_vec, local_offset)
        return local_offset + (q_w * t) + np.cross(q_vec, t)

    def _update_bearing_ema(self, bearing_x: float, bearing_y: float) -> np.ndarray:
        sample = np.array([bearing_x, bearing_y], dtype=np.float64)
        if not self._vision_ema_initialized:
            self._vision_bearing_ema = sample
            self._vision_ema_initialized = True
            return self._vision_bearing_ema.copy()

        alpha = float(np.clip(self.vision_bearing_ema_alpha, 0.0, 1.0))
        self._vision_bearing_ema = (
            alpha * sample + (1.0 - alpha) * self._vision_bearing_ema
        )
        return self._vision_bearing_ema.copy()

    def _current_search_radius_m(self) -> float:
        if not self.vision_enable_search or self.vision_search_radius_m <= 0.0:
            return 0.0

        radius = self.vision_search_radius_m
        if self._vision_lock_frames < self.vision_align_lock_frames:
            radius *= 1.0 + 0.15 * (
                self.vision_align_lock_frames - self._vision_lock_frames
            )
        if self._vision_lost_frames > 0:
            radius *= 1.0 + 0.25 * min(self._vision_lost_frames, 4)
        return float(min(radius, self.vision_search_max_radius_m))

    def _search_offset_local(self) -> np.ndarray:
        radius = self._current_search_radius_m()
        if radius <= 0.0:
            return np.zeros(3, dtype=np.float64)

        phase_elapsed = max(0.0, min(60.0, time.time() - self._phase_start_time))
        period = max(self.vision_search_period_s, 1.0e-3)
        theta = (2.0 * math.pi * phase_elapsed) / period
        return np.array(
            [
                radius * math.cos(theta),
                0.65 * radius * math.sin(2.0 * theta),
                0.0,
            ],
            dtype=np.float64,
        )

    def _build_guidance_context(self, observation_msg, task: Task):
        cv_observation = self._latest_structured_cv_observation
        if cv_observation is None:
            cv_observation = self._vision_pipeline.process_frame(
                raw_img=observation_msg.center_image,
                task_info=task,
                timestamp=self._cv_timestamp_s(),
                frame_id=self._vision_frame_id,
                camera="wrist",
            )
            self._vision_frame_id += 1
            self._latest_structured_cv_observation = cv_observation
            self._latest_cv_observation = cv_observation.to_harness_row()

        features = cv_observation.to_vision_target_features(
            image_width=float(max(1, observation_msg.center_image.width)),
            image_height=float(max(1, observation_msg.center_image.height)),
            valid_threshold=self.vision_valid_threshold,
        )
        selection_confidence = float(cv_observation.selection_confidence)
        valid = bool(features.target_valid >= 0.5 and not cv_observation.stale)
        segmentation_ok = (
            cv_observation.segmentation_confidence >= self.vision_min_segmentation_conf
        )
        selection_ok = selection_confidence >= self.vision_min_selection_conf
        scale_ok = cv_observation.target.scale_norm >= self.vision_min_scale_norm
        confidence = float(np.clip(cv_observation.target.confidence, 0.0, 1.0))
        acquired = valid and segmentation_ok and selection_ok

        if acquired:
            smoothed_bearing = self._update_bearing_ema(
                features.target_bearing_x,
                features.target_bearing_y,
            )
            self._vision_lock_frames = min(self._vision_lock_frames + 1, 10_000)
            self._vision_lost_frames = 0
            self._vision_last_locked_bearing = smoothed_bearing.copy()
            self._vision_last_locked_scale = float(features.target_scale_norm)
        else:
            self._vision_lost_frames += 1
            if self._vision_lost_frames > self.vision_loss_tolerance_frames:
                self._vision_lock_frames = 0
                self._vision_centered_frames = 0
            smoothed_bearing = (
                self._vision_last_locked_bearing.copy()
                if self._vision_ema_initialized
                else np.zeros(2, dtype=np.float64)
            )

        has_recent_lock = bool(
            self._vision_ema_initialized
            and (
                acquired
                or self._vision_lost_frames <= self.vision_loss_tolerance_frames
            )
        )
        bearing_x = float(smoothed_bearing[0])
        bearing_y = float(smoothed_bearing[1])
        centered = (
            has_recent_lock
            and abs(bearing_x) <= self.vision_align_bearing_tol_x
            and abs(bearing_y) <= self.vision_align_bearing_tol_y
        )

        if centered and (
            scale_ok or self._vision_last_locked_scale >= self.vision_min_scale_norm
        ):
            self._vision_centered_frames = min(self._vision_centered_frames + 1, 10_000)
        elif self._vision_lost_frames > self.vision_loss_tolerance_frames or acquired:
            self._vision_centered_frames = 0

        ready_for_align = has_recent_lock and (
            self._vision_lock_frames >= self.vision_align_lock_frames
        )
        ready_for_insert = ready_for_align and (
            self._vision_centered_frames >= self.vision_insert_centered_frames
        )
        allow_guarded_insert = ready_for_insert and (
            scale_ok or self._vision_last_locked_scale >= self.vision_min_scale_norm
        )
        if not self.vision_require_valid_for_insert:
            allow_guarded_insert = True

        guidance_context = {
            "cv_observation": cv_observation,
            "vision_mode": self.vision_mode,
            "advisory_only": self.vision_mode != "control",
            "features": features,
            "acquired": acquired,
            "has_recent_lock": has_recent_lock,
            "confidence": confidence,
            "segmentation_confidence": float(cv_observation.segmentation_confidence),
            "target_confidence": float(cv_observation.target.confidence),
            "selection_confidence": float(selection_confidence),
            "plug_confidence": float(cv_observation.plug.confidence),
            "scale_ok": scale_ok,
            "centered": centered,
            "ready_for_align": ready_for_align,
            "ready_for_insert": ready_for_insert,
            "allow_guarded_insert": allow_guarded_insert,
            "bearing_x": bearing_x,
            "bearing_y": bearing_y,
            "lock_frames": self._vision_lock_frames,
            "centered_frames": self._vision_centered_frames,
            "lost_frames": self._vision_lost_frames,
            "search_radius_m": self._current_search_radius_m(),
            "stale": bool(cv_observation.stale),
            "ambiguous": bool(cv_observation.ambiguous),
            "confidence_drop": bool(cv_observation.confidence_drop),
            "latency_ms": float(cv_observation.latency_ms),
            "guidance_applied": False,
            "guidance_would_apply": False,
            "proposed_local_offset_m": [0.0, 0.0, 0.0],
            "proposed_world_offset_m": [0.0, 0.0, 0.0],
        }
        self._last_guidance_context = guidance_context
        return guidance_context

    def _apply_guidance_pose(
        self,
        pose,
        observation_msg,
        task: Task,
        guidance_context,
    ):
        if guidance_context is None or self._phase not in {"APPROACH", "ALIGN"}:
            return pose

        guidance_pose = self._copy_pose(pose)
        tcp_pose = observation_msg.controller_state.tcp_pose
        local_offset = np.zeros(3, dtype=np.float64)

        if guidance_context["has_recent_lock"]:
            confidence_weight = float(
                np.clip(0.35 + (0.65 * guidance_context["confidence"]), 0.35, 1.0)
            )
            scale_weight = float(
                np.clip(
                    1.15 - (0.45 * guidance_context["features"].target_scale_norm),
                    0.65,
                    1.15,
                )
            )
            phase_weight = 0.45 if self._phase == "APPROACH" else 1.0
            if not guidance_context["ready_for_align"]:
                phase_weight *= 0.75
            local_offset += np.array(
                [
                    self.vision_x_sign
                    * guidance_context["bearing_x"]
                    * self.vision_bias_gain_x_m
                    * confidence_weight
                    * scale_weight
                    * phase_weight,
                    self.vision_y_sign
                    * guidance_context["bearing_y"]
                    * self.vision_bias_gain_y_m
                    * confidence_weight
                    * scale_weight
                    * phase_weight,
                    0.0,
                ],
                dtype=np.float64,
            )

        should_search = (
            not guidance_context["ready_for_align"]
            or not guidance_context["acquired"]
            or (self._phase == "ALIGN" and not guidance_context["centered"])
        )
        if should_search:
            search_weight = 0.35 if guidance_context["has_recent_lock"] else 1.0
            local_offset += search_weight * self._search_offset_local()

        offset_norm = float(np.linalg.norm(local_offset))
        combined_limit = max(
            self.vision_bias_limit_m,
            self.vision_search_max_radius_m,
        )
        if offset_norm > combined_limit > 0.0:
            local_offset *= combined_limit / offset_norm

        if np.linalg.norm(local_offset) <= 1.0e-9:
            guidance_context["guidance_would_apply"] = False
            guidance_context["guidance_applied"] = False
            return guidance_pose

        world_offset = self._rotate_local_offset(tcp_pose.orientation, local_offset)
        guidance_context["guidance_would_apply"] = True
        guidance_context["proposed_local_offset_m"] = [
            float(local_offset[0]),
            float(local_offset[1]),
            float(local_offset[2]),
        ]
        guidance_context["proposed_world_offset_m"] = [
            float(world_offset[0]),
            float(world_offset[1]),
            float(world_offset[2]),
        ]
        if self.vision_mode != "control":
            guidance_context["guidance_applied"] = False
            return guidance_pose

        guidance_pose.position.x += float(world_offset[0])
        guidance_pose.position.y += float(world_offset[1])
        guidance_pose.position.z += float(world_offset[2])
        guidance_context["guidance_applied"] = True
        return guidance_pose

    def _guidance_allows_align(self, guidance_context) -> bool:
        if guidance_context is None or self.vision_mode != "control":
            return True
        return bool(guidance_context["ready_for_align"])

    def _guidance_allows_guarded_insert(self, guidance_context) -> bool:
        if guidance_context is None or self.vision_mode != "control":
            return True
        return bool(guidance_context["allow_guarded_insert"])

    def _guidance_retry_phase(self, guidance_context) -> str:
        if guidance_context is None or self.vision_mode != "control":
            return "ALIGN"
        if (
            guidance_context["ready_for_align"]
            and guidance_context["has_recent_lock"]
            and guidance_context["confidence"] >= 0.65
        ):
            return "ALIGN"
        return "APPROACH"

    def _enter_retry(
        self,
        observation_msg,
        insert_direction: np.ndarray,
        now: float,
        reason: str,
    ) -> None:
        super()._enter_retry(observation_msg, insert_direction, now, reason)
        if self._retry_target_pose is None or self.vision_mode != "control":
            return

        guidance_context = self._last_guidance_context
        if guidance_context is None or not guidance_context["has_recent_lock"]:
            return

        current_pose = observation_msg.controller_state.tcp_pose
        local_offset = np.array(
            [
                self.vision_x_sign
                * guidance_context["bearing_x"]
                * self.vision_retry_bias_gain_x_m,
                self.vision_y_sign
                * guidance_context["bearing_y"]
                * self.vision_retry_bias_gain_y_m,
                0.0,
            ],
            dtype=np.float64,
        )
        offset_norm = float(np.linalg.norm(local_offset))
        retry_limit = max(
            self.vision_bias_limit_m,
            self.vision_retry_bias_gain_x_m,
            self.vision_retry_bias_gain_y_m,
        )
        if offset_norm > retry_limit > 0.0:
            local_offset *= retry_limit / offset_norm

        if np.linalg.norm(local_offset) <= 1.0e-9:
            return

        world_offset = self._rotate_local_offset(current_pose.orientation, local_offset)
        self._retry_target_pose.position.x += float(world_offset[0])
        self._retry_target_pose.position.y += float(world_offset[1])
        self._retry_target_pose.position.z += float(world_offset[2])

    def _guidance_summary(self, guidance_context) -> str:
        if guidance_context is None:
            return ""
        cv_observation = guidance_context["cv_observation"]
        features = guidance_context["features"]
        return (
            "vision="
            f"{guidance_context['vision_mode']} "
            f"{int(guidance_context['acquired'])} "
            f"tv={int(cv_observation.target.visible)} "
            f"pv={int(cv_observation.plug.visible)} "
            f"lock={guidance_context['lock_frames']}/{self.vision_align_lock_frames} "
            f"center={guidance_context['centered_frames']}/{self.vision_insert_centered_frames} "
            f"align={int(guidance_context['ready_for_align'])} "
            f"insert={int(guidance_context['ready_for_insert'])} "
            f"bx={guidance_context['bearing_x']:.2f} "
            f"by={guidance_context['bearing_y']:.2f} "
            f"terr=({cv_observation.visual_error.du:.1f},"
            f"{cv_observation.visual_error.dv:.1f},"
            f"{cv_observation.visual_error.dtheta_rad:.2f}) "
            f"sconf={guidance_context['segmentation_confidence']:.2f} "
            f"tconf={guidance_context['target_confidence']:.2f} "
            f"pconf={guidance_context['plug_confidence']:.2f} "
            f"sel={guidance_context['selection_confidence']:.2f} "
            f"scale={features.target_scale_norm:.2f} "
            f"stale={int(guidance_context['stale'])} "
            f"amb={int(guidance_context['ambiguous'])} "
            f"drop={int(guidance_context['confidence_drop'])} "
            f"apply={int(guidance_context['guidance_applied'])} "
            f"would={int(guidance_context['guidance_would_apply'])} "
            f"lat={guidance_context['latency_ms']:.1f}ms "
            f"lost={guidance_context['lost_frames']}"
        )
