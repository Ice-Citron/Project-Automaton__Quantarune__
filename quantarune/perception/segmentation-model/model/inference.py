from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from .model import load_port_segmenter_checkpoint
from .schema import task_condition_array_from_mapping


@dataclass(frozen=True, slots=True)
class MaskCandidate:
    score: float
    center_px: tuple[float, float]
    orientation_rad: float
    uncertainty_px: tuple[float, float]
    bbox_xyxy: tuple[float, float, float, float]
    scale_norm: float
    major_endpoints: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class PortSegmentationPrediction:
    target_prob: np.ndarray
    plug_prob: np.ndarray
    target_candidate: MaskCandidate | None
    plug_candidate: MaskCandidate | None
    selection_confidence: float


def _wrap_half_turn(angle_rad: float) -> float:
    while angle_rad > (0.5 * math.pi):
        angle_rad -= math.pi
    while angle_rad <= (-0.5 * math.pi):
        angle_rad += math.pi
    return angle_rad


def _component_candidates(
    prob_map: np.ndarray,
    *,
    threshold: float,
    min_area_frac: float,
) -> list[MaskCandidate]:
    if prob_map.ndim != 2:
        raise ValueError(f"Expected 2-D probability map, got {prob_map.shape!r}")

    height, width = prob_map.shape
    binary = (prob_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(height * width)

    candidates: list[MaskCandidate] = []
    for contour in contours:
        area_px = float(cv2.contourArea(contour))
        area_frac = area_px / max(image_area, 1.0)
        if area_frac < min_area_frac:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        component_mask = np.zeros_like(binary, dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, color=255, thickness=-1)
        component_scores = prob_map[component_mask > 0]
        if component_scores.size == 0:
            continue

        score = float(np.clip(component_scores.mean(), 0.0, 1.0))
        center_u = float(moments["m10"] / moments["m00"])
        center_v = float(moments["m01"] / moments["m00"])

        rect = cv2.minAreaRect(contour)
        rect_w = float(max(rect[1][0], 1.0))
        rect_h = float(max(rect[1][1], 1.0))
        angle_deg = float(rect[2] + (0.0 if rect_w >= rect_h else 90.0))
        angle_rad = _wrap_half_turn(math.radians(angle_deg))

        contour_points = contour.reshape(-1, 2).astype(np.float64)
        direction = np.array(
            [math.cos(angle_rad), math.sin(angle_rad)],
            dtype=np.float64,
        )
        projections = contour_points @ direction
        endpoint_a = contour_points[int(np.argmin(projections))]
        endpoint_b = contour_points[int(np.argmax(projections))]

        candidates.append(
            MaskCandidate(
                score=score,
                center_px=(center_u, center_v),
                orientation_rad=angle_rad,
                uncertainty_px=(float(w) / 2.0, float(h) / 2.0),
                bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                scale_norm=float(np.clip(area_frac / 0.05, 0.0, 1.0)),
                major_endpoints=(
                    (float(endpoint_a[0]), float(endpoint_a[1])),
                    (float(endpoint_b[0]), float(endpoint_b[1])),
                ),
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


class PortSegmenterPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        threshold: float = 0.35,
        min_area_frac: float = 0.0005,
    ) -> None:
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.min_area_frac = float(min_area_frac)

        model, payload = load_port_segmenter_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        model.eval()
        model.to(self.device)
        self.model = model
        self.payload = payload

    @torch.no_grad()
    def predict(
        self,
        image_rgb: np.ndarray,
        task_mapping: Mapping[str, Any] | None = None,
    ) -> PortSegmentationPrediction:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image shaped (H, W, 3), got {image_rgb.shape!r}")

        image = torch.from_numpy(np.transpose(image_rgb, (2, 0, 1)).copy()).float()
        image = image.unsqueeze(0).to(self.device) / 255.0
        task_condition = torch.from_numpy(
            task_condition_array_from_mapping(task_mapping)
        ).unsqueeze(0).to(self.device)

        outputs = self.model.predict_proba(image, task_condition)
        probs = outputs["mask_probs"][0].detach().cpu().numpy()
        target_prob = probs[0]
        plug_prob = probs[1]

        target_candidates = _component_candidates(
            target_prob,
            threshold=self.threshold,
            min_area_frac=self.min_area_frac,
        )
        plug_candidates = _component_candidates(
            plug_prob,
            threshold=self.threshold,
            min_area_frac=self.min_area_frac,
        )

        target_candidate = target_candidates[0] if target_candidates else None
        plug_candidate = plug_candidates[0] if plug_candidates else None

        if len(target_candidates) >= 2:
            top = target_candidates[0].score
            runner_up = target_candidates[1].score
            selection_confidence = float(np.clip(top - runner_up + 0.5, 0.0, 1.0))
        elif target_candidate is not None:
            selection_confidence = float(np.clip(target_candidate.score, 0.0, 1.0))
        else:
            selection_confidence = 0.0

        return PortSegmentationPrediction(
            target_prob=target_prob,
            plug_prob=plug_prob,
            target_candidate=target_candidate,
            plug_candidate=plug_candidate,
            selection_confidence=selection_confidence,
        )
