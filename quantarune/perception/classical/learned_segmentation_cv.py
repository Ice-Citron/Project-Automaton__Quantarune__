from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
TRAINING_ROOT = WORKSPACE_ROOT / "aic-quantarune-isaac-training" / "quantarune_training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from quantarune_training.vision import PortSegmentationPrediction, PortSegmenterPredictor

from .vision_target_features import (
    CVObservation,
    CVPlugObservation,
    CVTargetObservation,
    CVVisualError,
    VisionTargetFeatures,
    build_task_grounding_metadata,
    ros_image_to_rgb,
)


def _wrap_half_turn(angle_rad: float) -> float:
    while angle_rad > (0.5 * math.pi):
        angle_rad -= math.pi
    while angle_rad <= (-0.5 * math.pi):
        angle_rad += math.pi
    return angle_rad


def _build_task_mapping(task_msg: Any | None, task_hint: Mapping[str, Any] | None) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    if task_hint is not None:
        mapping.update(dict(task_hint))
    if task_msg is None:
        return mapping

    keys = (
        "cable_type_id",
        "plug_type_id",
        "port_type_id",
        "target_module_id",
        "target_port_id",
        "target_module_index",
        "target_port_index",
        "cable_type",
        "plug_type",
        "port_type",
        "port_name",
        "target_module_name",
        "task_text",
    )
    for key in keys:
        if hasattr(task_msg, key):
            mapping[key] = getattr(task_msg, key)
            mapping[f"task.{key}"] = getattr(task_msg, key)
    return mapping


def _choose_plug_tip(
    major_endpoints: tuple[tuple[float, float], tuple[float, float]],
    target_center: tuple[float, float] | None,
) -> tuple[float, float]:
    if target_center is None:
        return major_endpoints[0]
    ax, ay = major_endpoints[0]
    bx, by = major_endpoints[1]
    du_a = (ax - target_center[0]) ** 2 + (ay - target_center[1]) ** 2
    du_b = (bx - target_center[0]) ** 2 + (by - target_center[1]) ** 2
    return major_endpoints[0] if du_a <= du_b else major_endpoints[1]


def _candidate_to_target_obs(
    prediction: PortSegmentationPrediction,
    grounding: dict[str, Any],
) -> CVTargetObservation:
    candidate = prediction.target_candidate
    if candidate is None:
        return CVTargetObservation.empty(
            class_name=str(grounding.get("target_class", "target_port")),
            slot=str(grounding.get("target_slot", "unknown")),
        )
    return CVTargetObservation(
        visible=True,
        class_name=str(grounding.get("target_class", "target_port")),
        slot=str(grounding.get("target_slot", "unknown")),
        center_px=candidate.center_px,
        confidence=float(candidate.score),
        orientation_rad=float(candidate.orientation_rad),
        uncertainty_px=candidate.uncertainty_px,
        bbox_xyxy=candidate.bbox_xyxy,
        scale_norm=float(candidate.scale_norm),
    )


def _candidate_to_plug_obs(
    prediction: PortSegmentationPrediction,
    grounding: dict[str, Any],
    *,
    target_center: tuple[float, float] | None,
) -> CVPlugObservation:
    candidate = prediction.plug_candidate
    if candidate is None:
        return CVPlugObservation.empty(
            class_name=str(grounding.get("plug_class", "plug")),
        )
    return CVPlugObservation(
        visible=True,
        class_name=str(grounding.get("plug_class", "plug")),
        tip_px=_choose_plug_tip(candidate.major_endpoints, target_center),
        confidence=float(candidate.score),
        orientation_rad=float(candidate.orientation_rad),
        uncertainty_px=candidate.uncertainty_px,
        bbox_xyxy=candidate.bbox_xyxy,
        scale_norm=float(candidate.scale_norm),
    )


def _visual_error(
    target: CVTargetObservation,
    plug: CVPlugObservation,
) -> CVVisualError:
    if not target.visible or not plug.visible:
        return CVVisualError.zeros()
    return CVVisualError(
        du=float(plug.tip_px[0] - target.center_px[0]),
        dv=float(plug.tip_px[1] - target.center_px[1]),
        dtheta_rad=float(_wrap_half_turn(plug.orientation_rad - target.orientation_rad)),
    )


def _smooth_angle(prev_angle: float, new_angle: float, alpha: float) -> float:
    delta = _wrap_half_turn(new_angle - prev_angle)
    return _wrap_half_turn(prev_angle + alpha * delta)


def _smooth_target(
    previous: CVTargetObservation,
    current: CVTargetObservation,
    *,
    alpha: float,
) -> CVTargetObservation:
    if not previous.visible or not current.visible:
        return current
    return replace(
        current,
        center_px=(
            ((1.0 - alpha) * previous.center_px[0]) + (alpha * current.center_px[0]),
            ((1.0 - alpha) * previous.center_px[1]) + (alpha * current.center_px[1]),
        ),
        confidence=float(((1.0 - alpha) * previous.confidence) + (alpha * current.confidence)),
        orientation_rad=_smooth_angle(previous.orientation_rad, current.orientation_rad, alpha),
    )


def _smooth_plug(
    previous: CVPlugObservation,
    current: CVPlugObservation,
    *,
    alpha: float,
) -> CVPlugObservation:
    if not previous.visible or not current.visible:
        return current
    return replace(
        current,
        tip_px=(
            ((1.0 - alpha) * previous.tip_px[0]) + (alpha * current.tip_px[0]),
            ((1.0 - alpha) * previous.tip_px[1]) + (alpha * current.tip_px[1]),
        ),
        confidence=float(((1.0 - alpha) * previous.confidence) + (alpha * current.confidence)),
        orientation_rad=_smooth_angle(previous.orientation_rad, current.orientation_rad, alpha),
    )


def _observation_to_features(
    observation: CVObservation,
    *,
    width: int,
    height: int,
) -> VisionTargetFeatures:
    if not observation.target.visible:
        return VisionTargetFeatures.zeros()

    center_u = observation.target.center_px[0]
    center_v = observation.target.center_px[1]
    center_u_norm = float(np.clip(center_u / max(1.0, width - 1), 0.0, 1.0))
    center_v_norm = float(np.clip(center_v / max(1.0, height - 1), 0.0, 1.0))
    bearing_x = float(
        np.clip((center_u - (0.5 * (width - 1))) / max(1.0, 0.5 * (width - 1)), -1.0, 1.0)
    )
    bearing_y = float(
        np.clip((center_v - (0.5 * (height - 1))) / max(1.0, 0.5 * (height - 1)), -1.0, 1.0)
    )

    return VisionTargetFeatures(
        target_valid=1.0,
        target_center_u_norm=center_u_norm,
        target_center_v_norm=center_v_norm,
        target_scale_norm=float(np.clip(observation.target.scale_norm, 0.0, 1.0)),
        target_bearing_x=bearing_x,
        target_bearing_y=bearing_y,
        segmentation_confidence=float(np.clip(observation.segmentation_confidence, 0.0, 1.0)),
        selection_confidence=float(np.clip(observation.selection_confidence, 0.0, 1.0)),
    )


class LearnedSegmentationCVPipeline:
    """Learned segmentation-backed CV pipeline with the existing CVObservation API."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        camera: str = "wrist",
        threshold: float = 0.35,
        stale_after_s: float = 0.25,
        smoothing_alpha: float = 0.65,
    ) -> None:
        self.camera = camera
        self.stale_after_s = float(stale_after_s)
        self.smoothing_alpha = float(smoothing_alpha)
        self.predictor = PortSegmenterPredictor(
            checkpoint_path,
            device=device,
            threshold=threshold,
        )
        self._latest_observation: CVObservation | None = None

    def process_frame(
        self,
        frame,
        task_info: Any | None = None,
        *,
        timestamp: float | None = None,
        frame_id: int | None = None,
    ) -> CVObservation:
        t0 = time.perf_counter()
        image_rgb = frame if isinstance(frame, np.ndarray) else ros_image_to_rgb(frame)
        timestamp_value = float(timestamp if timestamp is not None else time.time())
        frame_id_value = int(frame_id if frame_id is not None else 0)

        task_msg = None if isinstance(task_info, Mapping) else task_info
        task_hint = dict(task_info) if isinstance(task_info, Mapping) else None
        task_mapping = _build_task_mapping(task_msg, task_hint)
        grounding = build_task_grounding_metadata(task_msg=task_msg, task_hint=task_hint)
        prediction = self.predictor.predict(image_rgb, task_mapping)

        target = _candidate_to_target_obs(prediction, grounding)
        plug = _candidate_to_plug_obs(
            prediction,
            grounding,
            target_center=target.center_px if target.visible else None,
        )

        ambiguous = bool(target.visible and prediction.selection_confidence < 0.60)
        confidence_drop = False
        stale = False
        if self._latest_observation is not None:
            confidence_drop = bool(
                self._latest_observation.target.visible
                and target.visible
                and (self._latest_observation.target.confidence - target.confidence) > 0.15
            )
            target = _smooth_target(
                self._latest_observation.target,
                target,
                alpha=self.smoothing_alpha,
            )
            plug = _smooth_plug(
                self._latest_observation.plug,
                plug,
                alpha=self.smoothing_alpha,
            )

        observation = CVObservation(
            timestamp=timestamp_value,
            frame_id=frame_id_value,
            camera=self.camera,
            target=target,
            plug=plug,
            visual_error=_visual_error(target, plug),
            stale=False,
            latency_ms=float((time.perf_counter() - t0) * 1000.0),
            ambiguous=ambiguous,
            confidence_drop=confidence_drop,
            segmentation_confidence=float(target.confidence),
            selection_confidence=float(np.clip(prediction.selection_confidence, 0.0, 1.0)),
        )

        if (
            not observation.target.visible
            and self._latest_observation is not None
            and (timestamp_value - self._latest_observation.timestamp) <= self.stale_after_s
        ):
            stale = True
            observation = replace(
                self._latest_observation,
                timestamp=timestamp_value,
                frame_id=frame_id_value,
                stale=True,
                latency_ms=float((time.perf_counter() - t0) * 1000.0),
                confidence_drop=True,
            )
        else:
            observation = replace(observation, stale=stale)

        self._latest_observation = observation
        return observation

    def extract(
        self,
        *,
        raw_img=None,
        image_rgb: np.ndarray | None = None,
        task_msg: Any | None = None,
        task_hint: Mapping[str, Any] | None = None,
    ) -> VisionTargetFeatures:
        if image_rgb is None:
            if raw_img is None:
                return VisionTargetFeatures.zeros()
            image_rgb = ros_image_to_rgb(raw_img)

        if image_rgb is None or image_rgb.size == 0:
            return VisionTargetFeatures.zeros()

        observation = self.process_frame(
            image_rgb,
            task_hint if task_hint is not None else task_msg,
        )
        height, width = image_rgb.shape[:2]
        return _observation_to_features(observation, width=width, height=height)

    def get_latest_observation(self) -> CVObservation | None:
        return self._latest_observation

    def get_latest_harness_row(self) -> dict[str, Any] | None:
        if self._latest_observation is None:
            return None
        return self._latest_observation.to_harness_row()


class LearnedSegmentationFeatureExtractor:
    """Feature-only adapter for swapping learned masks into the ACT56 feature seam."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        threshold: float = 0.35,
    ) -> None:
        self.pipeline = LearnedSegmentationCVPipeline(
            checkpoint_path,
            device=device,
            threshold=threshold,
        )

    def extract(
        self,
        *,
        raw_img=None,
        image_rgb: np.ndarray | None = None,
        task_msg: Any | None = None,
        task_hint: Mapping[str, Any] | None = None,
    ) -> VisionTargetFeatures:
        return self.pipeline.extract(
            raw_img=raw_img,
            image_rgb=image_rgb,
            task_msg=task_msg,
            task_hint=task_hint,
        )
