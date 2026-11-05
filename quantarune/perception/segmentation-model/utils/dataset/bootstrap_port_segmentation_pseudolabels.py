from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
RUNTIME_POLICY_ROOT = WORKSPACE_ROOT / "aic-quantarune" / "aic_example_policies"
if str(RUNTIME_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_POLICY_ROOT))

from aic_example_policies.ros.vision_target_features import CenterPortCVPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap weak target / plug segmentation masks from the current "
            "heuristic center-camera CV pipeline."
        )
    )
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument(
        "--metadata-jsonl",
        default=None,
        help="Optional path to the annotation metadata JSONL. Defaults to annotation-root/metadata.jsonl.",
    )
    parser.add_argument(
        "--target-mask-root",
        default="masks/target",
        help="Relative output root for target masks inside the annotation pack.",
    )
    parser.add_argument(
        "--plug-mask-root",
        default="masks/plug",
        help="Relative output root for plug masks inside the annotation pack.",
    )
    parser.add_argument("--min-target-confidence", type=float, default=0.20)
    parser.add_argument("--min-plug-confidence", type=float, default=0.15)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _contour_mask_for_candidate(
    mask: np.ndarray,
    candidate,
) -> np.ndarray:
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_score = float("inf")
    target_center = np.array([candidate.center_u, candidate.center_v], dtype=np.float64)

    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        center = np.array(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float64,
        )
        x, y, w, h = cv2.boundingRect(contour)
        bbox = np.array([x, y, x + w, y + h], dtype=np.float64)
        candidate_bbox = np.array(candidate.bbox_xyxy, dtype=np.float64)
        center_dist = float(np.linalg.norm(center - target_center))
        bbox_dist = float(np.linalg.norm(bbox - candidate_bbox))
        score = center_dist + (0.10 * bbox_dist)
        if score < best_score:
            best_score = score
            best_contour = contour

    output = np.zeros((height, width), dtype=np.uint8)
    if best_contour is not None:
        cv2.drawContours(output, [best_contour], contourIdx=-1, color=255, thickness=-1)
    return output


def main() -> int:
    args = parse_args()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    metadata_path = (
        Path(args.metadata_jsonl).expanduser().resolve()
        if args.metadata_jsonl is not None
        else annotation_root / "metadata.jsonl"
    )

    rows = _load_jsonl(metadata_path)
    if not rows:
        raise SystemExit(f"No metadata rows found in {metadata_path}")

    pipeline = CenterPortCVPipeline()
    target_root = annotation_root / args.target_mask_root
    plug_root = annotation_root / args.plug_mask_root
    target_root.mkdir(parents=True, exist_ok=True)
    plug_root.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    for row in rows:
        image_rel = Path(str(row["image_path"]))
        image_abs = annotation_root / image_rel
        image_rgb = _read_rgb(image_abs)

        task_hint = dict(row)
        mask = pipeline._extractor._build_mask(image_rgb)
        candidates = pipeline._extractor._extract_candidates(mask)

        target_candidate = None
        selection_conf = 0.0
        target_confidence = 0.0
        if candidates:
            target_candidate, selection_conf = pipeline._extractor._select_candidate(
                candidates,
                task_msg=None,
                task_hint=task_hint,
            )
            target_confidence = float(
                np.clip(
                    math.sqrt(
                        max(target_candidate.score, 0.0) * max(selection_conf, 0.0)
                    ),
                    0.0,
                    1.0,
                )
            )

        plug_candidate = None
        plug_confidence = 0.0
        if candidates:
            plug_candidate, plug_confidence = pipeline._select_plug_candidate(
                candidates,
                target_candidate=target_candidate,
            )

        target_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        if target_candidate is not None and target_confidence >= args.min_target_confidence:
            target_mask = _contour_mask_for_candidate(mask, target_candidate)
            counts["target_mask_written"] += 1
        else:
            counts["target_mask_empty"] += 1

        plug_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        if plug_candidate is not None and plug_confidence >= args.min_plug_confidence:
            plug_mask = _contour_mask_for_candidate(mask, plug_candidate)
            counts["plug_mask_written"] += 1
        else:
            counts["plug_mask_empty"] += 1

        target_out = target_root / image_rel.relative_to("images")
        plug_out = plug_root / image_rel.relative_to("images")
        target_out.parent.mkdir(parents=True, exist_ok=True)
        plug_out.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(target_out), target_mask)
        cv2.imwrite(str(plug_out), plug_mask)
        counts["rows"] += 1

    summary = {
        "annotation_root": str(annotation_root),
        "metadata_path": str(metadata_path),
        "rows": int(counts["rows"]),
        "target_mask_written": int(counts["target_mask_written"]),
        "target_mask_empty": int(counts["target_mask_empty"]),
        "plug_mask_written": int(counts["plug_mask_written"]),
        "plug_mask_empty": int(counts["plug_mask_empty"]),
        "target_mask_root": args.target_mask_root,
        "plug_mask_root": args.plug_mask_root,
        "min_target_confidence": float(args.min_target_confidence),
        "min_plug_confidence": float(args.min_plug_confidence),
    }
    (annotation_root / "pseudolabel_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
