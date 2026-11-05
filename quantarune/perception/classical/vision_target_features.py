from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from aic_task_interfaces.msg import Task


VISION_FEATURE_NAMES = (
    "vision.target_valid",
    "vision.target_center_u_norm",
    "vision.target_center_v_norm",
    "vision.target_scale_norm",
    "vision.target_bearing_x",
    "vision.target_bearing_y",
    "vision.segmentation_confidence",
    "vision.selection_confidence",
)

TASK_TEXT_PATTERN = re.compile(
    r"insert\s+(?P<cable_type>\S+)\s+cable\s+\S+\s+with\s+"
    r"(?P<plug_type>\S+)\s+plug\s+\S+\s+into\s+"
    r"(?P<port_type>\S+)\s+port\s+(?P<port_name>\S+)\s+on\s+module\s+"
    r"(?P<target_module_name>\S+)",
    flags=re.IGNORECASE,
)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value is not None else float(default)


@dataclass(frozen=True, slots=True)
class VisionTargetFeatures:
    target_valid: float
    target_center_u_norm: float
    target_center_v_norm: float
    target_scale_norm: float
    target_bearing_x: float
    target_bearing_y: float
    segmentation_confidence: float
    selection_confidence: float

    @classmethod
    def zeros(cls) -> "VisionTargetFeatures":
        return cls(
            target_valid=0.0,
            target_center_u_norm=0.0,
            target_center_v_norm=0.0,
            target_scale_norm=0.0,
            target_bearing_x=0.0,
            target_bearing_y=0.0,
            segmentation_confidence=0.0,
            selection_confidence=0.0,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "vision.target_valid": self.target_valid,
            "vision.target_center_u_norm": self.target_center_u_norm,
            "vision.target_center_v_norm": self.target_center_v_norm,
            "vision.target_scale_norm": self.target_scale_norm,
            "vision.target_bearing_x": self.target_bearing_x,
            "vision.target_bearing_y": self.target_bearing_y,
            "vision.segmentation_confidence": self.segmentation_confidence,
            "vision.selection_confidence": self.selection_confidence,
        }

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.target_valid,
                self.target_center_u_norm,
                self.target_center_v_norm,
                self.target_scale_norm,
                self.target_bearing_x,
                self.target_bearing_y,
                self.segmentation_confidence,
                self.selection_confidence,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    score: float
    center_u: float
    center_v: float
    area_frac: float
    fill_ratio: float
    aspect_ratio: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_wh: tuple[float, float]
    angle_rad: float
    uncertainty_px: tuple[float, float]
    major_endpoints: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class CVTargetObservation:
    visible: bool
    class_name: str
    slot: str
    center_px: tuple[float, float]
    confidence: float
    orientation_rad: float
    uncertainty_px: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    scale_norm: float

    @classmethod
    def empty(
        cls,
        *,
        class_name: str = "target",
        slot: str = "unknown",
    ) -> "CVTargetObservation":
        return cls(
            visible=False,
            class_name=class_name,
            slot=slot,
            center_px=(0.0, 0.0),
            confidence=0.0,
            orientation_rad=0.0,
            uncertainty_px=(0.0, 0.0),
            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            scale_norm=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": bool(self.visible),
            "class": self.class_name,
            "slot": self.slot,
            "center_px": [float(self.center_px[0]), float(self.center_px[1])],
            "confidence": float(self.confidence),
            "orientation_rad": float(self.orientation_rad),
            "uncertainty_px": [
                float(self.uncertainty_px[0]),
                float(self.uncertainty_px[1]),
            ],
            "bbox": [
                float(self.bbox_xyxy[0]),
                float(self.bbox_xyxy[1]),
                float(self.bbox_xyxy[2]),
                float(self.bbox_xyxy[3]),
            ],
            "scale_norm": float(self.scale_norm),
        }


@dataclass(frozen=True, slots=True)
class CVPlugObservation:
    visible: bool
    class_name: str
    tip_px: tuple[float, float]
    confidence: float
    orientation_rad: float
    uncertainty_px: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    scale_norm: float

    @classmethod
    def empty(cls, *, class_name: str = "plug") -> "CVPlugObservation":
        return cls(
            visible=False,
            class_name=class_name,
            tip_px=(0.0, 0.0),
            confidence=0.0,
            orientation_rad=0.0,
            uncertainty_px=(0.0, 0.0),
            bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
            scale_norm=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible": bool(self.visible),
            "class": self.class_name,
            "tip_px": [float(self.tip_px[0]), float(self.tip_px[1])],
            "confidence": float(self.confidence),
            "orientation_rad": float(self.orientation_rad),
            "uncertainty_px": [
                float(self.uncertainty_px[0]),
                float(self.uncertainty_px[1]),
            ],
            "bbox": [
                float(self.bbox_xyxy[0]),
                float(self.bbox_xyxy[1]),
                float(self.bbox_xyxy[2]),
                float(self.bbox_xyxy[3]),
            ],
            "scale_norm": float(self.scale_norm),
        }


@dataclass(frozen=True, slots=True)
class CVVisualError:
    du: float
    dv: float
    dtheta_rad: float

    @classmethod
    def zeros(cls) -> "CVVisualError":
        return cls(du=0.0, dv=0.0, dtheta_rad=0.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "du": float(self.du),
            "dv": float(self.dv),
            "dtheta_rad": float(self.dtheta_rad),
        }


@dataclass(frozen=True, slots=True)
class CVPoseEstimate:
    valid: bool
    method: str
    target_translation_camera_m: tuple[float, float, float]
    lateral_error_m: tuple[float, float]
    yaw_error_rad: float
    range_m: float
    reprojection_error_px: float
    confidence: float
    reason: str

    @classmethod
    def invalid(cls, reason: str) -> "CVPoseEstimate":
        return cls(
            valid=False,
            method="unavailable",
            target_translation_camera_m=(0.0, 0.0, 0.0),
            lateral_error_m=(0.0, 0.0),
            yaw_error_rad=0.0,
            range_m=0.0,
            reprojection_error_px=float("inf"),
            confidence=0.0,
            reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "method": self.method,
            "target_translation_camera_m": [
                float(self.target_translation_camera_m[0]),
                float(self.target_translation_camera_m[1]),
                float(self.target_translation_camera_m[2]),
            ],
            "lateral_error_m": [
                float(self.lateral_error_m[0]),
                float(self.lateral_error_m[1]),
            ],
            "yaw_error_rad": float(self.yaw_error_rad),
            "range_m": float(self.range_m),
            "reprojection_error_px": float(self.reprojection_error_px),
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CVObservation:
    timestamp: float
    frame_id: int
    camera: str
    target: CVTargetObservation
    plug: CVPlugObservation
    visual_error: CVVisualError
    stale: bool
    latency_ms: float
    ambiguous: bool
    confidence_drop: bool
    segmentation_confidence: float
    selection_confidence: float
    pose_estimate: CVPoseEstimate | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "frame_id": int(self.frame_id),
            "camera": self.camera,
            "target": self.target.to_dict(),
            "plug": self.plug.to_dict(),
            "visual_error": self.visual_error.to_dict(),
            "stale": bool(self.stale),
            "latency_ms": float(self.latency_ms),
            "ambiguous": bool(self.ambiguous),
            "confidence_drop": bool(self.confidence_drop),
            "segmentation_confidence": float(self.segmentation_confidence),
            "selection_confidence": float(self.selection_confidence),
            "pose_estimate": (
                self.pose_estimate.to_dict()
                if self.pose_estimate is not None
                else CVPoseEstimate.invalid("not_computed").to_dict()
            ),
            "detections": self._to_detections(),
        }

    def to_harness_row(
        self,
        *,
        phase: str | None = None,
        stage: str | None = None,
        supervisor_phase: str | None = None,
    ) -> dict[str, Any]:
        row = self.to_dict()
        if phase:
            row["phase"] = phase
        if stage:
            row["stage"] = stage
        if supervisor_phase:
            row["supervisor_phase"] = supervisor_phase
        return row

    def _to_detections(self) -> list[dict[str, Any]]:
        detections: list[dict[str, Any]] = []
        if self.target.visible:
            detections.append(
                {
                    "class": self.target.class_name,
                    "bbox": [
                        float(self.target.bbox_xyxy[0]),
                        float(self.target.bbox_xyxy[1]),
                        float(self.target.bbox_xyxy[2]),
                        float(self.target.bbox_xyxy[3]),
                    ],
                    "confidence": float(self.target.confidence),
                    "orientation_rad": float(self.target.orientation_rad),
                    "uncertainty_px": [
                        float(self.target.uncertainty_px[0]),
                        float(self.target.uncertainty_px[1]),
                    ],
                    "slot": self.target.slot,
                    "keypoints": {
                        "center": [
                            float(self.target.center_px[0]),
                            float(self.target.center_px[1]),
                        ]
                    },
                }
            )
        if self.plug.visible:
            detections.append(
                {
                    "class": self.plug.class_name,
                    "bbox": [
                        float(self.plug.bbox_xyxy[0]),
                        float(self.plug.bbox_xyxy[1]),
                        float(self.plug.bbox_xyxy[2]),
                        float(self.plug.bbox_xyxy[3]),
                    ],
                    "confidence": float(self.plug.confidence),
                    "orientation_rad": float(self.plug.orientation_rad),
                    "uncertainty_px": [
                        float(self.plug.uncertainty_px[0]),
                        float(self.plug.uncertainty_px[1]),
                    ],
                    "keypoints": {
                        "tip": [float(self.plug.tip_px[0]), float(self.plug.tip_px[1])]
                    },
                }
            )
        return detections

    def to_vision_target_features(
        self,
        *,
        image_width: float,
        image_height: float,
        valid_threshold: float,
    ) -> VisionTargetFeatures:
        center_u_norm = 0.0
        center_v_norm = 0.0
        bearing_x = 0.0
        bearing_y = 0.0
        if image_width > 1.0 and image_height > 1.0:
            center_u_norm = float(
                np.clip(self.target.center_px[0] / max(1.0, image_width - 1.0), 0.0, 1.0)
            )
            center_v_norm = float(
                np.clip(self.target.center_px[1] / max(1.0, image_height - 1.0), 0.0, 1.0)
            )
            bearing_x = float(
                np.clip(
                    (self.target.center_px[0] - (0.5 * (image_width - 1.0)))
                    / max(1.0, 0.5 * (image_width - 1.0)),
                    -1.0,
                    1.0,
                )
            )
            bearing_y = float(
                np.clip(
                    (self.target.center_px[1] - (0.5 * (image_height - 1.0)))
                    / max(1.0, 0.5 * (image_height - 1.0)),
                    -1.0,
                    1.0,
                )
            )

        target_valid = (
            1.0
            if self.target.visible
            and not self.stale
            and self.segmentation_confidence >= float(valid_threshold)
            else 0.0
        )
        return VisionTargetFeatures(
            target_valid=target_valid,
            target_center_u_norm=center_u_norm,
            target_center_v_norm=center_v_norm,
            target_scale_norm=float(np.clip(self.target.scale_norm, 0.0, 1.0)),
            target_bearing_x=bearing_x,
            target_bearing_y=bearing_y,
            segmentation_confidence=float(
                np.clip(self.segmentation_confidence, 0.0, 1.0)
            ),
            selection_confidence=float(np.clip(self.selection_confidence, 0.0, 1.0)),
        )


class ClassicalPortPoseEstimator:
    """Estimate live plug-to-port geometry from center-camera keypoints.

    This is deliberately conservative: it does not require privileged TF and it
    falls back to calibrated-FOV pinhole geometry when camera_info is absent.
    A tiny diagonal Kalman update smooths the local lateral/yaw/range estimate
    without hiding large jumps behind an overfit replay template.
    """

    def __init__(
        self,
        *,
        horizontal_fov_deg: float | None = None,
        vertical_fov_deg: float | None = None,
        min_confidence: float | None = None,
        max_reprojection_error_px: float | None = None,
    ):
        self._horizontal_fov_deg = float(
            horizontal_fov_deg
            if horizontal_fov_deg is not None
            else _env_float("AIC_CLASSICAL_CENTER_HFOV_DEG", 70.0)
        )
        self._vertical_fov_deg = float(
            vertical_fov_deg
            if vertical_fov_deg is not None
            else _env_float("AIC_CLASSICAL_CENTER_VFOV_DEG", 55.0)
        )
        self._min_confidence = float(
            min_confidence
            if min_confidence is not None
            else _env_float("AIC_CLASSICAL_POSE_MIN_CONFIDENCE", 0.12)
        )
        self._max_reprojection_error_px = float(
            max_reprojection_error_px
            if max_reprojection_error_px is not None
            else _env_float("AIC_CLASSICAL_POSE_MAX_REPROJECTION_PX", 45.0)
        )
        self._range_min_m = _env_float("AIC_CLASSICAL_POSE_MIN_RANGE_M", 0.04)
        self._range_max_m = _env_float("AIC_CLASSICAL_POSE_MAX_RANGE_M", 0.55)
        self._process_noise = np.array([4.0e-6, 4.0e-6, 2.0e-4, 4.0e-5], dtype=np.float64)
        self._measurement_noise = np.array([2.5e-5, 2.5e-5, 2.0e-3, 2.5e-4], dtype=np.float64)
        self._state: np.ndarray | None = None
        self._covariance: np.ndarray | None = None

    def reset(self) -> None:
        self._state = None
        self._covariance = None

    def estimate(
        self,
        *,
        target: CVTargetObservation,
        plug: CVPlugObservation,
        visual_error: CVVisualError,
        image_width: int,
        image_height: int,
    ) -> CVPoseEstimate:
        if image_width <= 1 or image_height <= 1:
            return CVPoseEstimate.invalid("invalid_image_size")
        if not target.visible:
            return CVPoseEstimate.invalid("target_not_visible")

        bbox_width_px = max(1.0, float(target.bbox_xyxy[2] - target.bbox_xyxy[0]))
        bbox_height_px = max(1.0, float(target.bbox_xyxy[3] - target.bbox_xyxy[1]))
        fx, fy, cx, cy = self._camera_intrinsics(image_width, image_height)
        physical_width_m, physical_height_m = self._target_dimensions_m(target.class_name)
        range_from_width = fx * physical_width_m / bbox_width_px
        range_from_height = fy * physical_height_m / bbox_height_px
        range_m = float(
            np.clip(
                0.5 * (range_from_width + range_from_height),
                self._range_min_m,
                self._range_max_m,
            )
        )

        tx_m = ((float(target.center_px[0]) - cx) / max(fx, 1.0e-6)) * range_m
        ty_m = ((float(target.center_px[1]) - cy) / max(fy, 1.0e-6)) * range_m
        target_translation = (float(tx_m), float(ty_m), float(range_m))

        if not plug.visible:
            confidence = 0.5 * float(target.confidence)
            return CVPoseEstimate(
                valid=confidence >= self._min_confidence,
                method="bbox_range_target_only",
                target_translation_camera_m=target_translation,
                lateral_error_m=(0.0, 0.0),
                yaw_error_rad=0.0,
                range_m=range_m,
                reprojection_error_px=0.5 * (float(target.uncertainty_px[0]) + float(target.uncertainty_px[1])),
                confidence=confidence,
                reason="plug_not_visible",
            )

        lateral_x_m = (float(visual_error.du) / max(fx, 1.0e-6)) * range_m
        lateral_y_m = (float(visual_error.dv) / max(fy, 1.0e-6)) * range_m
        yaw_error_rad = float(_wrap_half_turn(visual_error.dtheta_rad))
        confidence = float(
            np.clip(
                math.sqrt(max(target.confidence, 0.0) * max(plug.confidence, 0.0)),
                0.0,
                1.0,
            )
        )
        reprojection_error_px = self._pseudo_reprojection_error_px(
            target=target,
            plug=plug,
            range_from_width=range_from_width,
            range_from_height=range_from_height,
        )
        valid = bool(
            confidence >= self._min_confidence
            and reprojection_error_px <= self._max_reprojection_error_px
        )
        if not valid:
            reason = (
                "low_confidence"
                if confidence < self._min_confidence
                else "high_reprojection_error"
            )
            return CVPoseEstimate.invalid(reason)

        filtered = self._kalman_update(
            np.array([lateral_x_m, lateral_y_m, yaw_error_rad, range_m], dtype=np.float64),
            confidence=confidence,
        )
        return CVPoseEstimate(
            valid=True,
            method="bbox_range_keypoint_kalman",
            target_translation_camera_m=target_translation,
            lateral_error_m=(float(filtered[0]), float(filtered[1])),
            yaw_error_rad=float(filtered[2]),
            range_m=float(filtered[3]),
            reprojection_error_px=float(reprojection_error_px),
            confidence=confidence,
            reason="ok",
        )

    def _camera_intrinsics(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        fx_env = os.environ.get("AIC_CLASSICAL_CENTER_FX_PX")
        fy_env = os.environ.get("AIC_CLASSICAL_CENTER_FY_PX")
        cx_env = os.environ.get("AIC_CLASSICAL_CENTER_CX_PX")
        cy_env = os.environ.get("AIC_CLASSICAL_CENTER_CY_PX")
        fx = (
            float(fx_env)
            if fx_env is not None
            else float(image_width) / (2.0 * math.tan(math.radians(self._horizontal_fov_deg) * 0.5))
        )
        fy = (
            float(fy_env)
            if fy_env is not None
            else float(image_height) / (2.0 * math.tan(math.radians(self._vertical_fov_deg) * 0.5))
        )
        cx = float(cx_env) if cx_env is not None else 0.5 * float(image_width - 1)
        cy = float(cy_env) if cy_env is not None else 0.5 * float(image_height - 1)
        return fx, fy, cx, cy

    @staticmethod
    def _target_dimensions_m(class_name: str) -> tuple[float, float]:
        normalized = (class_name or "").lower()
        if "sfp" in normalized:
            return (
                _env_float("AIC_CLASSICAL_SFP_PORT_WIDTH_M", 0.016),
                _env_float("AIC_CLASSICAL_SFP_PORT_HEIGHT_M", 0.010),
            )
        if "lc" in normalized:
            return (
                _env_float("AIC_CLASSICAL_LC_PORT_WIDTH_M", 0.010),
                _env_float("AIC_CLASSICAL_LC_PORT_HEIGHT_M", 0.007),
            )
        return (
            _env_float("AIC_CLASSICAL_SC_PORT_WIDTH_M", 0.012),
            _env_float("AIC_CLASSICAL_SC_PORT_HEIGHT_M", 0.008),
        )

    @staticmethod
    def _pseudo_reprojection_error_px(
        *,
        target: CVTargetObservation,
        plug: CVPlugObservation,
        range_from_width: float,
        range_from_height: float,
    ) -> float:
        size_consistency_px = abs(range_from_width - range_from_height) * 20.0
        uncertainty_px = 0.25 * (
            float(target.uncertainty_px[0])
            + float(target.uncertainty_px[1])
            + float(plug.uncertainty_px[0])
            + float(plug.uncertainty_px[1])
        )
        return float(max(0.0, size_consistency_px + uncertainty_px))

    def _kalman_update(self, measurement: np.ndarray, *, confidence: float) -> np.ndarray:
        if self._state is None or self._covariance is None:
            self._state = measurement.astype(np.float64)
            self._covariance = np.array([1.0e-4, 1.0e-4, 1.0e-2, 1.0e-3], dtype=np.float64)
            return self._state.copy()

        self._covariance = self._covariance + self._process_noise
        noise_scale = 1.0 / max(float(confidence) * float(confidence), 0.05)
        measurement_noise = self._measurement_noise * noise_scale
        kalman_gain = self._covariance / (self._covariance + measurement_noise)
        innovation = measurement - self._state
        innovation[2] = _wrap_half_turn(float(innovation[2]))
        self._state = self._state + (kalman_gain * innovation)
        self._state[2] = _wrap_half_turn(float(self._state[2]))
        self._covariance = (1.0 - kalman_gain) * self._covariance
        return self._state.copy()


@dataclass(slots=True)
class _TrackState:
    visible: bool = False
    center_px: tuple[float, float] = (0.0, 0.0)
    orientation_rad: float = 0.0
    confidence: float = 0.0
    uncertainty_px: tuple[float, float] = (0.0, 0.0)
    bbox_xyxy: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    scale_norm: float = 0.0
    timestamp: float = float("-inf")


def _extract_suffix_index(name: str) -> int:
    match = re.search(r"(\d+)$", name or "")
    return int(match.group(1)) if match else -1


def _wrap_half_turn(angle_rad: float) -> float:
    wrapped = (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -0.5 * math.pi:
        wrapped += math.pi
    elif wrapped > 0.5 * math.pi:
        wrapped -= math.pi
    return float(wrapped)


def _name_prefix_token(name: str | None) -> str | None:
    if not isinstance(name, str) or not name:
        return None
    token = name.split("_", 1)[0].strip().lower()
    return token or None


def _angle_delta(target_rad: float, source_rad: float) -> float:
    return _wrap_half_turn(float(target_rad) - float(source_rad))


def _candidate_scale_norm(
    candidate: _Candidate,
    *,
    min_area_frac: float,
    max_area_frac: float,
) -> float:
    return float(
        np.clip(
            math.sqrt(candidate.area_frac / max(min_area_frac, 1e-8))
            / math.sqrt(max_area_frac / max(min_area_frac, 1e-8)),
            0.0,
            1.0,
        )
    )


def _task_value(task_msg: Any | None, task_hint: dict[str, Any] | None, key: str) -> Any:
    if task_hint is not None and key in task_hint:
        return task_hint[key]
    if task_msg is not None and hasattr(task_msg, key):
        return getattr(task_msg, key)
    return None


def parse_task_text(task_text: str | None) -> dict[str, Any]:
    if not isinstance(task_text, str) or not task_text.strip():
        return {}
    match = TASK_TEXT_PATTERN.search(task_text.strip())
    if match is None:
        return {"task_text": task_text}

    payload = match.groupdict()
    payload["task_text"] = task_text
    payload["target_module_index"] = _extract_suffix_index(
        payload.get("target_module_name", "")
    )
    payload["target_port_index"] = _extract_suffix_index(payload.get("port_name", ""))
    return payload


def _slot_name(task_msg: Any | None, task_hint: dict[str, Any] | None) -> str:
    module_name = _task_value(task_msg, task_hint, "target_module_name")
    if isinstance(module_name, str) and module_name:
        return module_name

    module_index = _task_value(task_msg, task_hint, "task.target_module_index")
    if module_index is None:
        module_index = _extract_suffix_index(
            str(_task_value(task_msg, task_hint, "target_module_name") or "")
        )
    try:
        module_index_int = int(float(module_index))
    except (TypeError, ValueError):
        module_index_int = -1
    if module_index_int >= 0:
        return f"nic_card_mount_{module_index_int}"
    return "unknown"


def _port_class_name(task_msg: Any | None, task_hint: dict[str, Any] | None) -> str:
    port_type = _task_value(task_msg, task_hint, "port_type")
    if isinstance(port_type, str) and port_type:
        return f"{port_type}_port"

    port_type_id = _task_value(task_msg, task_hint, "task.port_type_id")
    port_type_map = {1: "sc_port", 2: "sfp_port"}
    try:
        if port_type_id is not None:
            mapped = port_type_map.get(int(float(port_type_id)))
            if mapped is not None:
                return mapped
    except (TypeError, ValueError):
        pass
    inferred = _name_prefix_token(
        str(_task_value(task_msg, task_hint, "port_name") or "")
    )
    if inferred is not None:
        return f"{inferred}_port"
    return "target_port"


def _plug_class_name(task_msg: Any | None, task_hint: dict[str, Any] | None) -> str:
    plug_type = _task_value(task_msg, task_hint, "plug_type")
    if isinstance(plug_type, str) and plug_type:
        return f"{plug_type}_plug"

    plug_type_id = _task_value(task_msg, task_hint, "task.plug_type_id")
    plug_type_map = {1: "sc_plug", 2: "sfp_plug", 3: "lc_plug"}
    try:
        if plug_type_id is not None:
            mapped = plug_type_map.get(int(float(plug_type_id)))
            if mapped is not None:
                return mapped
    except (TypeError, ValueError):
        pass
    inferred = _name_prefix_token(str(_task_value(task_msg, task_hint, "plug_name") or ""))
    if inferred is not None:
        return f"{inferred}_plug"
    return "plug"


def build_task_grounding_metadata(
    *,
    task_msg: Any | None = None,
    task_hint: dict[str, Any] | None = None,
    task_text: str | None = None,
    episode_id: str | None = None,
    frame_idx: int | None = None,
    success: bool | None = None,
    failure_type: str | None = None,
) -> dict[str, Any]:
    parsed_text = parse_task_text(task_text)

    port_name_value = _task_value(task_msg, task_hint, "port_name")
    if not isinstance(port_name_value, str) or not port_name_value:
        port_name_value = parsed_text.get("port_name") or "unknown"

    target_module_name = _task_value(task_msg, task_hint, "target_module_name")
    if not isinstance(target_module_name, str) or not target_module_name:
        target_module_name = parsed_text.get("target_module_name") or _slot_name(
            task_msg, task_hint
        )

    cable_type = _task_value(task_msg, task_hint, "cable_type")
    if not isinstance(cable_type, str) or not cable_type:
        cable_type = parsed_text.get("cable_type") or "unknown"

    plug_type = _task_value(task_msg, task_hint, "plug_type")
    if not isinstance(plug_type, str) or not plug_type:
        plug_type = parsed_text.get("plug_type")
    if not isinstance(plug_type, str) or not plug_type:
        plug_type = _plug_class_name(task_msg, task_hint).removesuffix("_plug")

    port_type = _task_value(task_msg, task_hint, "port_type")
    if not isinstance(port_type, str) or not port_type:
        port_type = parsed_text.get("port_type")
    if not isinstance(port_type, str) or not port_type:
        port_type = _port_class_name(task_msg, task_hint).removesuffix("_port")

    target_module_index = _task_value(task_msg, task_hint, "task.target_module_index")
    if target_module_index is None and task_hint is not None:
        target_module_index = task_hint.get("target_module_index")
    if target_module_index is None:
        target_module_index = parsed_text.get("target_module_index")
    if target_module_index is None:
        target_module_index = _extract_suffix_index(target_module_name)

    target_port_index = _task_value(task_msg, task_hint, "task.target_port_index")
    if target_port_index is None and task_hint is not None:
        target_port_index = task_hint.get("target_port_index")
    if target_port_index is None:
        target_port_index = parsed_text.get("target_port_index")
    if target_port_index is None:
        target_port_index = _extract_suffix_index(port_name_value)

    try:
        target_module_index = int(float(target_module_index))
    except (TypeError, ValueError):
        target_module_index = -1

    try:
        target_port_index = int(float(target_port_index))
    except (TypeError, ValueError):
        target_port_index = -1

    task_text_value = task_text if isinstance(task_text, str) and task_text else None
    if task_text_value is None and parsed_text.get("task_text"):
        task_text_value = str(parsed_text["task_text"])

    return {
        "task_text": task_text_value,
        "cable_class": cable_type,
        "plug_class": f"{plug_type}_plug" if plug_type != "unknown" else "plug",
        "target_class": f"{port_type}_port" if port_type != "unknown" else "target_port",
        "target_slot": target_module_name,
        "target_module_name": target_module_name,
        "target_module_index": target_module_index,
        "target_port_index": target_port_index,
        "port_name": port_name_value,
        "episode_id": episode_id,
        "frame_idx": frame_idx,
        "success": success,
        "failure_type": failure_type,
    }


def ros_image_to_rgb(raw_img) -> np.ndarray:
    channels = 4 if raw_img.encoding in {"rgba8", "bgra8"} else 3
    image_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
        raw_img.height, raw_img.width, channels
    )

    if raw_img.encoding == "rgb8":
        return image_np
    if raw_img.encoding == "bgr8":
        return image_np[:, :, ::-1]
    if raw_img.encoding == "rgba8":
        return image_np[:, :, :3]
    if raw_img.encoding == "bgra8":
        return image_np[:, :, :3][:, :, ::-1]

    # Fall back to assuming RGB layout when the encoding is omitted.
    return image_np[:, :, :3]


class CenterPortFeatureExtractor:
    """Extract compact target-port features from the center RGB image.

    The implementation is intentionally lightweight and deterministic so the
    feature contract can stabilize before a learned segmenter replaces it.
    """

    def __init__(
        self,
        *,
        min_area_frac: float = 2e-4,
        max_area_frac: float = 1.5e-1,
        valid_threshold: float = 0.35,
    ):
        self._min_area_frac = float(min_area_frac)
        self._max_area_frac = float(max_area_frac)
        self._valid_threshold = float(valid_threshold)

    def extract(
        self,
        *,
        raw_img=None,
        image_rgb: np.ndarray | None = None,
        task_msg: Any | None = None,
        task_hint: dict[str, float] | None = None,
    ) -> VisionTargetFeatures:
        if image_rgb is None:
            if raw_img is None:
                return VisionTargetFeatures.zeros()
            image_rgb = ros_image_to_rgb(raw_img)

        if image_rgb is None or image_rgb.size == 0:
            return VisionTargetFeatures.zeros()

        height, width = image_rgb.shape[:2]
        if height <= 1 or width <= 1:
            return VisionTargetFeatures.zeros()

        mask = self._build_mask(image_rgb)
        candidates = self._extract_candidates(mask)
        if not candidates:
            return VisionTargetFeatures.zeros()

        candidate, selection_conf = self._select_candidate(
            candidates,
            task_msg=task_msg,
            task_hint=task_hint,
        )
        target_valid = 1.0 if candidate.score >= self._valid_threshold else 0.0
        center_u_norm = float(np.clip(candidate.center_u / max(1.0, width - 1), 0.0, 1.0))
        center_v_norm = float(np.clip(candidate.center_v / max(1.0, height - 1), 0.0, 1.0))
        scale_norm = _candidate_scale_norm(
            candidate,
            min_area_frac=self._min_area_frac,
            max_area_frac=self._max_area_frac,
        )
        bearing_x = float(
            np.clip((candidate.center_u - (0.5 * (width - 1))) / max(1.0, 0.5 * (width - 1)), -1.0, 1.0)
        )
        bearing_y = float(
            np.clip((candidate.center_v - (0.5 * (height - 1))) / max(1.0, 0.5 * (height - 1)), -1.0, 1.0)
        )

        return VisionTargetFeatures(
            target_valid=target_valid,
            target_center_u_norm=center_u_norm,
            target_center_v_norm=center_v_norm,
            target_scale_norm=scale_norm,
            target_bearing_x=bearing_x,
            target_bearing_y=bearing_y,
            segmentation_confidence=float(np.clip(candidate.score, 0.0, 1.0)),
            selection_confidence=float(np.clip(selection_conf, 0.0, 1.0)),
        )

    def _build_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        dark_mask = cv2.inRange(
            hsv,
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((180, 255, 110), dtype=np.uint8),
        )
        mask = cv2.bitwise_and(adaptive, dark_mask)

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _extract_candidates(self, mask: np.ndarray) -> list[_Candidate]:
        height, width = mask.shape[:2]
        image_area = float(height * width)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[_Candidate] = []
        for contour in contours:
            area_px = float(cv2.contourArea(contour))
            area_frac = area_px / max(image_area, 1.0)
            if area_frac < self._min_area_frac or area_frac > self._max_area_frac:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            rect_area = float(max(1, w * h))
            fill_ratio = area_px / rect_area
            if fill_ratio < 0.2:
                continue

            aspect_ratio = float(w) / float(max(1, h))
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0.0:
                continue

            rect = cv2.minAreaRect(contour)
            rect_w = float(max(rect[1][0], 1.0))
            rect_h = float(max(rect[1][1], 1.0))
            angle_deg = float(rect[2] + (0.0 if rect_w >= rect_h else 90.0))
            angle_rad = _wrap_half_turn(math.radians(angle_deg))
            contour_points = contour.reshape(-1, 2).astype(np.float64)
            direction = np.array(
                [math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64
            )
            projections = contour_points @ direction
            endpoint_a = contour_points[int(np.argmin(projections))]
            endpoint_b = contour_points[int(np.argmax(projections))]

            center_u = moments["m10"] / moments["m00"]
            center_v = moments["m01"] / moments["m00"]
            center_dx = (center_u - (0.5 * (width - 1))) / max(1.0, 0.5 * (width - 1))
            center_dy = (center_v - (0.5 * (height - 1))) / max(1.0, 0.5 * (height - 1))
            center_distance = math.sqrt((center_dx * center_dx) + (center_dy * center_dy))
            center_score = max(0.0, 1.0 - min(1.0, center_distance))
            fill_score = max(0.0, min(1.0, fill_ratio / 0.75))
            area_score = max(0.0, min(1.0, area_frac / 0.02))
            aspect_score = max(
                0.0,
                1.0 - min(1.0, abs(math.log(max(aspect_ratio, 1e-6))) / math.log(4.0)),
            )

            score = (
                0.35 * center_score
                + 0.25 * fill_score
                + 0.20 * area_score
                + 0.20 * aspect_score
            )
            candidates.append(
                _Candidate(
                    score=float(score),
                    center_u=float(center_u),
                    center_v=float(center_v),
                    area_frac=float(area_frac),
                    fill_ratio=float(fill_ratio),
                    aspect_ratio=float(aspect_ratio),
                    bbox_xyxy=(
                        float(x),
                        float(y),
                        float(x + w),
                        float(y + h),
                    ),
                    bbox_wh=(float(w), float(h)),
                    angle_rad=float(angle_rad),
                    uncertainty_px=(
                        float(1.0 + (0.5 * max(1.0, w) * (1.0 - score))),
                        float(1.0 + (0.5 * max(1.0, h) * (1.0 - score))),
                    ),
                    major_endpoints=(
                        (float(endpoint_a[0]), float(endpoint_a[1])),
                        (float(endpoint_b[0]), float(endpoint_b[1])),
                    ),
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _select_candidate(
        self,
        candidates: list[_Candidate],
        *,
        task_msg: Any | None,
        task_hint: dict[str, float] | None,
    ) -> tuple[_Candidate, float]:
        if len(candidates) == 1:
            return candidates[0], 1.0

        desired_index = self._desired_candidate_index(task_msg, task_hint)
        if desired_index is not None and 0 <= desired_index < len(candidates):
            ordered = sorted(candidates, key=lambda item: item.center_u)
            chosen = ordered[desired_index]
            score_gap = max(0.0, min(1.0, chosen.score - ordered[0].score + 0.5))
            return chosen, max(0.45, score_gap)

        top = candidates[0]
        runner_up = candidates[1]
        gap = max(0.0, min(1.0, 0.5 + (top.score - runner_up.score)))
        return top, gap

    @staticmethod
    def _desired_candidate_index(
        task_msg: Any | None,
        task_hint: dict[str, float] | None,
    ) -> int | None:
        if task_hint is not None:
            port_index = int(task_hint.get("task.target_port_index", -1.0))
            if port_index >= 0:
                return port_index

            module_index = int(task_hint.get("task.target_module_index", -1.0))
            if module_index >= 0:
                return module_index

        if task_msg is None:
            return None

        port_index = _extract_suffix_index(task_msg.port_name)
        if port_index >= 0:
            return port_index

        module_index = _extract_suffix_index(task_msg.target_module_name)
        if module_index >= 0:
            return module_index

        return None


class CenterPortCVPipeline:
    """Stateful CV pipeline that emits control-relevant, harness-ready observations."""

    def __init__(
        self,
        *,
        extractor: CenterPortFeatureExtractor | None = None,
        camera_name: str = "wrist",
        tracking_alpha: float = 0.40,
        jump_alpha: float = 0.15,
        max_tracking_jump_px: float = 80.0,
        stale_after_s: float = 0.25,
        plug_min_confidence: float = 0.20,
    ):
        self._extractor = extractor or CenterPortFeatureExtractor()
        self._camera_name = camera_name
        self._tracking_alpha = float(tracking_alpha)
        self._jump_alpha = float(jump_alpha)
        self._max_tracking_jump_px = float(max_tracking_jump_px)
        self._stale_after_s = float(stale_after_s)
        self._plug_min_confidence = float(plug_min_confidence)
        self._target_track = _TrackState()
        self._plug_track = _TrackState()
        self._pose_estimator = ClassicalPortPoseEstimator()
        self._latest_observation: CVObservation | None = None
        self._frame_counter = -1

    def reset(self) -> None:
        self._target_track = _TrackState()
        self._plug_track = _TrackState()
        self._pose_estimator.reset()
        self._latest_observation = None
        self._frame_counter = -1

    def process_frame(
        self,
        frame=None,
        task_info: Any | None = None,
        *,
        raw_img=None,
        image_rgb: np.ndarray | None = None,
        task_msg: Any | None = None,
        task_hint: dict[str, Any] | None = None,
        timestamp: float | None = None,
        frame_id: int | None = None,
        camera: str | None = None,
    ) -> CVObservation:
        start = time.perf_counter()
        if frame is not None:
            if image_rgb is None and isinstance(frame, np.ndarray):
                image_rgb = frame
            elif raw_img is None:
                raw_img = frame

        if task_info is not None:
            if isinstance(task_info, dict):
                task_hint = task_info
            else:
                task_msg = task_info

        if image_rgb is None:
            if raw_img is None:
                raise ValueError("process_frame requires frame, raw_img, or image_rgb")
            image_rgb = ros_image_to_rgb(raw_img)

        if timestamp is None:
            timestamp = time.time()
        if frame_id is None:
            self._frame_counter += 1
            frame_id = self._frame_counter

        height, width = image_rgb.shape[:2]
        mask = self._extractor._build_mask(image_rgb)
        candidates = self._extractor._extract_candidates(mask)

        target_class = _port_class_name(task_msg, task_hint)
        plug_class = _plug_class_name(task_msg, task_hint)
        slot_name = _slot_name(task_msg, task_hint)

        target_candidate = None
        selection_conf = 0.0
        segmentation_conf = 0.0
        if candidates:
            target_candidate, selection_conf = self._extractor._select_candidate(
                candidates,
                task_msg=task_msg,
                task_hint=task_hint,
            )
            segmentation_conf = float(np.clip(target_candidate.score, 0.0, 1.0))
        target_confidence = 0.0
        if target_candidate is not None:
            target_confidence = float(
                np.clip(
                    math.sqrt(max(segmentation_conf, 0.0) * max(selection_conf, 0.0)),
                    0.0,
                    1.0,
                )
            )

        plug_candidate, plug_confidence = self._select_plug_candidate(
            candidates,
            target_candidate=target_candidate,
        )

        target_visible_raw = (
            target_candidate is not None
            and target_candidate.score >= self._extractor._valid_threshold
        )
        plug_visible_raw = (
            plug_candidate is not None and plug_confidence >= self._plug_min_confidence
        )

        target_obs, target_stale, target_conf_drop = self._track_target(
            candidate=target_candidate,
            visible_raw=target_visible_raw,
            confidence=target_confidence,
            timestamp=float(timestamp),
            class_name=target_class,
            slot=slot_name,
        )
        plug_obs, plug_stale, plug_conf_drop = self._track_plug(
            candidate=plug_candidate,
            target_center_px=target_obs.center_px,
            visible_raw=plug_visible_raw,
            confidence=plug_confidence,
            timestamp=float(timestamp),
            class_name=plug_class,
        )

        ambiguous = bool(
            len(candidates) > 1
            and (selection_conf < 0.65 or (candidates[0].score - candidates[1].score) < 0.08)
        )
        stale = bool(target_stale or plug_stale)
        confidence_drop = bool(target_conf_drop or plug_conf_drop)

        if target_obs.visible and plug_obs.visible:
            visual_error = CVVisualError(
                du=float(plug_obs.tip_px[0] - target_obs.center_px[0]),
                dv=float(plug_obs.tip_px[1] - target_obs.center_px[1]),
                dtheta_rad=float(
                    _angle_delta(plug_obs.orientation_rad, target_obs.orientation_rad)
                ),
            )
        else:
            visual_error = CVVisualError.zeros()

        pose_estimate = self._pose_estimator.estimate(
            target=target_obs,
            plug=plug_obs,
            visual_error=visual_error,
            image_width=int(width),
            image_height=int(height),
        )

        observation = CVObservation(
            timestamp=float(timestamp),
            frame_id=int(frame_id),
            camera=camera or self._camera_name,
            target=target_obs,
            plug=plug_obs,
            visual_error=visual_error,
            stale=stale,
            latency_ms=float((time.perf_counter() - start) * 1000.0),
            ambiguous=ambiguous,
            confidence_drop=confidence_drop,
            segmentation_confidence=segmentation_conf,
            selection_confidence=float(np.clip(selection_conf, 0.0, 1.0)),
            pose_estimate=pose_estimate,
        )
        self._latest_observation = observation
        return observation

    def get_latest_observation(self) -> CVObservation | None:
        return self._latest_observation

    def get_latest_harness_row(self) -> dict[str, Any] | None:
        if self._latest_observation is None:
            return None
        return self._latest_observation.to_harness_row()

    def _select_plug_candidate(
        self,
        candidates: list[_Candidate],
        *,
        target_candidate: _Candidate | None,
    ) -> tuple[_Candidate | None, float]:
        remaining = [
            candidate
            for candidate in candidates
            if target_candidate is None or candidate.bbox_xyxy != target_candidate.bbox_xyxy
        ]
        if not remaining:
            return None, 0.0

        target_center = None
        if target_candidate is not None:
            target_center = np.array(
                [target_candidate.center_u, target_candidate.center_v], dtype=np.float64
            )

        best_candidate = None
        best_score = float("-inf")
        second_score = float("-inf")
        for candidate in remaining:
            candidate_score = candidate.score
            if target_center is not None:
                distance_px = float(
                    np.linalg.norm(
                        np.array([candidate.center_u, candidate.center_v], dtype=np.float64)
                        - target_center
                    )
                )
                proximity_bonus = max(0.0, 1.0 - min(1.0, distance_px / 220.0))
                candidate_score += 0.10 * proximity_bonus
            elongation_bonus = min(
                1.0, abs(math.log(max(candidate.aspect_ratio, 1.0e-6))) / math.log(6.0)
            )
            candidate_score += 0.08 * elongation_bonus

            if candidate_score > best_score:
                second_score = best_score
                best_score = candidate_score
                best_candidate = candidate
            elif candidate_score > second_score:
                second_score = candidate_score

        if best_candidate is None:
            return None, 0.0

        gap = 1.0 if second_score == float("-inf") else max(0.0, min(1.0, 0.5 + (best_score - second_score)))
        confidence = float(np.clip(math.sqrt(max(best_candidate.score, 0.0) * gap), 0.0, 1.0))
        return best_candidate, confidence

    def _smooth_track(
        self,
        track: _TrackState,
        *,
        center_px: tuple[float, float],
        orientation_rad: float,
        confidence: float,
        uncertainty_px: tuple[float, float],
        bbox_xyxy: tuple[float, float, float, float],
        scale_norm: float,
        timestamp: float,
    ) -> tuple[_TrackState, bool]:
        confidence_drop = False
        if not track.visible:
            updated = _TrackState(
                visible=True,
                center_px=center_px,
                orientation_rad=orientation_rad,
                confidence=confidence,
                uncertainty_px=uncertainty_px,
                bbox_xyxy=bbox_xyxy,
                scale_norm=scale_norm,
                timestamp=timestamp,
            )
            return updated, confidence_drop

        jump_px = math.dist(center_px, track.center_px)
        alpha = self._tracking_alpha if jump_px <= self._max_tracking_jump_px else self._jump_alpha
        smoothed_center = (
            float((alpha * center_px[0]) + ((1.0 - alpha) * track.center_px[0])),
            float((alpha * center_px[1]) + ((1.0 - alpha) * track.center_px[1])),
        )
        smoothed_orientation = _wrap_half_turn(
            track.orientation_rad + (alpha * _angle_delta(orientation_rad, track.orientation_rad))
        )
        smoothed_confidence = float(
            np.clip((alpha * confidence) + ((1.0 - alpha) * track.confidence), 0.0, 1.0)
        )
        confidence_drop = confidence < max(0.0, track.confidence - 0.20)
        smoothed_uncertainty = (
            float((alpha * uncertainty_px[0]) + ((1.0 - alpha) * track.uncertainty_px[0])),
            float((alpha * uncertainty_px[1]) + ((1.0 - alpha) * track.uncertainty_px[1])),
        )
        smoothed_bbox = tuple(
            float((alpha * curr) + ((1.0 - alpha) * prev))
            for curr, prev in zip(bbox_xyxy, track.bbox_xyxy)
        )
        updated = _TrackState(
            visible=True,
            center_px=smoothed_center,
            orientation_rad=smoothed_orientation,
            confidence=smoothed_confidence,
            uncertainty_px=smoothed_uncertainty,
            bbox_xyxy=smoothed_bbox,
            scale_norm=float((alpha * scale_norm) + ((1.0 - alpha) * track.scale_norm)),
            timestamp=timestamp,
        )
        return updated, confidence_drop

    def _stale_track(self, track: _TrackState, *, timestamp: float) -> _TrackState | None:
        if not track.visible:
            return None
        age_s = float(timestamp - track.timestamp)
        if age_s < 0.0 or age_s > self._stale_after_s:
            return None
        decay = max(0.0, 1.0 - (age_s / max(self._stale_after_s, 1.0e-6)))
        return _TrackState(
            visible=True,
            center_px=track.center_px,
            orientation_rad=track.orientation_rad,
            confidence=float(np.clip(track.confidence * decay, 0.0, 1.0)),
            uncertainty_px=(
                float(track.uncertainty_px[0] + (4.0 * age_s)),
                float(track.uncertainty_px[1] + (4.0 * age_s)),
            ),
            bbox_xyxy=track.bbox_xyxy,
            scale_norm=track.scale_norm,
            timestamp=track.timestamp,
        )

    def _track_target(
        self,
        *,
        candidate: _Candidate | None,
        visible_raw: bool,
        confidence: float,
        timestamp: float,
        class_name: str,
        slot: str,
    ) -> tuple[CVTargetObservation, bool, bool]:
        confidence_drop = False
        stale = False
        if candidate is not None and visible_raw:
            scale_norm = _candidate_scale_norm(
                candidate,
                min_area_frac=self._extractor._min_area_frac,
                max_area_frac=self._extractor._max_area_frac,
            )
            self._target_track, confidence_drop = self._smooth_track(
                self._target_track,
                center_px=(candidate.center_u, candidate.center_v),
                orientation_rad=candidate.angle_rad,
                confidence=confidence,
                uncertainty_px=candidate.uncertainty_px,
                bbox_xyxy=candidate.bbox_xyxy,
                scale_norm=scale_norm,
                timestamp=timestamp,
            )
        else:
            carried = self._stale_track(self._target_track, timestamp=timestamp)
            if carried is None:
                self._target_track = _TrackState()
            else:
                self._target_track = carried
                stale = True

        if not self._target_track.visible:
            return CVTargetObservation.empty(class_name=class_name, slot=slot), stale, confidence_drop

        observation = CVTargetObservation(
            visible=True,
            class_name=class_name,
            slot=slot,
            center_px=self._target_track.center_px,
            confidence=self._target_track.confidence,
            orientation_rad=self._target_track.orientation_rad,
            uncertainty_px=self._target_track.uncertainty_px,
            bbox_xyxy=self._target_track.bbox_xyxy,
            scale_norm=self._target_track.scale_norm,
        )
        return observation, stale, confidence_drop

    def _track_plug(
        self,
        *,
        candidate: _Candidate | None,
        target_center_px: tuple[float, float],
        visible_raw: bool,
        confidence: float,
        timestamp: float,
        class_name: str,
    ) -> tuple[CVPlugObservation, bool, bool]:
        confidence_drop = False
        stale = False
        if candidate is not None and visible_raw:
            tip_px = self._plug_tip(candidate, target_center_px)
            scale_norm = _candidate_scale_norm(
                candidate,
                min_area_frac=self._extractor._min_area_frac,
                max_area_frac=self._extractor._max_area_frac,
            )
            self._plug_track, confidence_drop = self._smooth_track(
                self._plug_track,
                center_px=tip_px,
                orientation_rad=candidate.angle_rad,
                confidence=confidence,
                uncertainty_px=candidate.uncertainty_px,
                bbox_xyxy=candidate.bbox_xyxy,
                scale_norm=scale_norm,
                timestamp=timestamp,
            )
        else:
            carried = self._stale_track(self._plug_track, timestamp=timestamp)
            if carried is None:
                self._plug_track = _TrackState()
            else:
                self._plug_track = carried
                stale = True

        if not self._plug_track.visible:
            return CVPlugObservation.empty(class_name=class_name), stale, confidence_drop

        observation = CVPlugObservation(
            visible=True,
            class_name=class_name,
            tip_px=self._plug_track.center_px,
            confidence=self._plug_track.confidence,
            orientation_rad=self._plug_track.orientation_rad,
            uncertainty_px=self._plug_track.uncertainty_px,
            bbox_xyxy=self._plug_track.bbox_xyxy,
            scale_norm=self._plug_track.scale_norm,
        )
        return observation, stale, confidence_drop

    @staticmethod
    def _plug_tip(
        candidate: _Candidate,
        target_center_px: tuple[float, float],
    ) -> tuple[float, float]:
        endpoint_a, endpoint_b = candidate.major_endpoints
        if target_center_px == (0.0, 0.0):
            return endpoint_a if endpoint_a[1] <= endpoint_b[1] else endpoint_b
        distance_a = math.dist(endpoint_a, target_center_px)
        distance_b = math.dist(endpoint_b, target_center_px)
        return endpoint_a if distance_a <= distance_b else endpoint_b
