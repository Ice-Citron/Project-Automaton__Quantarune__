from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from quantarune_training.vision import PortSegmenterPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a high-value manual-correction subset for the learned port "
            "segmentation lane using model uncertainty and pseudo-label disagreement."
        )
    )
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-mask-root", default="masks/target")
    parser.add_argument("--plug-mask-root", default="masks/plug")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--max-per-slot", type=int, default=24)
    parser.add_argument("--per-criterion", type=int, default=8)
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


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (mask > 127).astype(np.uint8)


def _binary_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    union = np.logical_or(pred_b, target_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred_b, target_b).sum()
    return float(intersection / union)


def _norm_center(center: tuple[float, float] | None) -> list[float] | None:
    if center is None:
        return None
    return [float(center[0]), float(center[1])]


def _draw_mask_contours(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_bgr, contours, -1, color, 1)


def _render_overlay(
    image_rgb: np.ndarray,
    *,
    target_mask: np.ndarray,
    plug_mask: np.ndarray,
    target_bbox: tuple[float, float, float, float] | None,
    plug_bbox: tuple[float, float, float, float] | None,
    target_center: tuple[float, float] | None,
    plug_tip: tuple[float, float] | None,
    stats: dict[str, Any],
    output_path: Path,
) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _draw_mask_contours(image_bgr, target_mask, (255, 0, 0))
    _draw_mask_contours(image_bgr, plug_mask, (0, 255, 255))

    if target_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in target_bbox]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if plug_bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in plug_bbox]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)

    if target_center is not None:
        cv2.circle(
            image_bgr,
            (int(round(target_center[0])), int(round(target_center[1]))),
            4,
            (0, 0, 255),
            -1,
        )
    if plug_tip is not None:
        cv2.circle(
            image_bgr,
            (int(round(plug_tip[0])), int(round(plug_tip[1]))),
            4,
            (0, 200, 0),
            -1,
        )

    lines = [
        f"slot={stats['target_slot']} frame={stats['frame_idx']}",
        f"seg={stats['segmentation_confidence']:.3f} sel={stats['selection_confidence']:.3f}",
        f"iou_t={stats['target_iou_vs_pseudo']:.3f} iou_p={stats['plug_iou_vs_pseudo']:.3f}",
        f"scale={stats['target_scale_norm']:.3f} review={stats['review_reasons']}",
    ]
    y = 18
    for line in lines:
        cv2.putText(
            image_bgr,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 18

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def main() -> int:
    args = parse_args()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    predictor = PortSegmenterPredictor(
        checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    rows = _load_jsonl(annotation_root / "metadata.jsonl")

    target_mask_root = annotation_root / args.target_mask_root
    plug_mask_root = annotation_root / args.plug_mask_root

    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        image_path = annotation_root / row["image_path"]
        image_rgb = _read_rgb(image_path)
        prediction = predictor.predict(image_rgb, row)

        pred_target_mask = (prediction.target_prob >= args.threshold).astype(np.uint8)
        pred_plug_mask = (prediction.plug_prob >= args.threshold).astype(np.uint8)

        rel_no_images = Path(row["image_path"]).relative_to("images")
        pseudo_target_mask = _read_mask(target_mask_root / rel_no_images)
        pseudo_plug_mask = _read_mask(plug_mask_root / rel_no_images)

        target_iou = _binary_iou(pred_target_mask, pseudo_target_mask)
        plug_iou = _binary_iou(pred_plug_mask, pseudo_plug_mask)

        target_candidate = prediction.target_candidate
        plug_candidate = prediction.plug_candidate
        segmentation_confidence = float(target_candidate.score) if target_candidate else 0.0
        selection_confidence = float(prediction.selection_confidence)
        target_scale_norm = float(target_candidate.scale_norm) if target_candidate else 0.0

        scored_rows.append(
            {
                **row,
                "target_iou_vs_pseudo": target_iou,
                "plug_iou_vs_pseudo": plug_iou,
                "segmentation_confidence": segmentation_confidence,
                "selection_confidence": selection_confidence,
                "target_scale_norm": target_scale_norm,
                "target_center_px": _norm_center(
                    target_candidate.center_px if target_candidate else None
                ),
                "plug_tip_px": _norm_center(
                    plug_candidate.major_endpoints[0] if plug_candidate else None
                ),
                "target_bbox": list(target_candidate.bbox_xyxy) if target_candidate else None,
                "plug_bbox": list(plug_candidate.bbox_xyxy) if plug_candidate else None,
            }
        )

    by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_slot[str(row.get("target_slot", "unknown"))].append(row)

    selected_rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, int]] = set()
    criteria = (
        ("low_selection_confidence", lambda item: item["selection_confidence"]),
        ("low_segmentation_confidence", lambda item: item["segmentation_confidence"]),
        ("low_target_iou", lambda item: item["target_iou_vs_pseudo"]),
        ("large_target_scale", lambda item: -item["target_scale_norm"]),
    )

    for slot, slot_rows in sorted(by_slot.items()):
        slot_selected: list[dict[str, Any]] = []
        reason_map: dict[tuple[str, int], list[str]] = defaultdict(list)

        for reason, key_fn in criteria:
            ordered = sorted(slot_rows, key=key_fn)
            take_count = 0
            for row in ordered:
                key = (str(row["episode_id"]), int(row["frame_idx"]))
                reason_map[key].append(reason)
                if key not in selected_keys:
                    slot_selected.append(row)
                    selected_keys.add(key)
                    take_count += 1
                if take_count >= args.per_criterion:
                    break

        slot_selected = slot_selected[: args.max_per_slot]
        for row in slot_selected:
            key = (str(row["episode_id"]), int(row["frame_idx"]))
            row = dict(row)
            row["review_reasons"] = sorted(set(reason_map[key]))
            selected_rows.append(row)

    selected_rows.sort(key=lambda item: (str(item["target_slot"]), str(item["episode_id"]), int(item["frame_idx"])))

    review_manifest_path = output_root / "review_manifest.jsonl"
    review_csv_path = output_root / "review_manifest.csv"
    overlays_root = output_root / "overlays"
    images_root = output_root / "images"
    masks_root = output_root / "masks"

    with review_manifest_path.open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    import csv

    fieldnames = sorted({key for row in selected_rows for key in row.keys()})
    with review_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    slot_counts: dict[str, int] = Counter(str(row["target_slot"]) for row in selected_rows)
    for row in selected_rows:
        rel_no_images = Path(row["image_path"]).relative_to("images")
        image_src = annotation_root / row["image_path"]
        target_mask_src = target_mask_root / rel_no_images
        plug_mask_src = plug_mask_root / rel_no_images
        image_rgb = _read_rgb(image_src)
        target_mask = _read_mask(target_mask_src)
        plug_mask = _read_mask(plug_mask_src)

        image_dst = images_root / rel_no_images
        target_mask_dst = masks_root / "target" / rel_no_images
        plug_mask_dst = masks_root / "plug" / rel_no_images
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        target_mask_dst.parent.mkdir(parents=True, exist_ok=True)
        plug_mask_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_src, image_dst)
        shutil.copy2(target_mask_src, target_mask_dst)
        shutil.copy2(plug_mask_src, plug_mask_dst)

        overlay_path = overlays_root / rel_no_images
        _render_overlay(
            image_rgb,
            target_mask=target_mask,
            plug_mask=plug_mask,
            target_bbox=tuple(row["target_bbox"]) if row["target_bbox"] is not None else None,
            plug_bbox=tuple(row["plug_bbox"]) if row["plug_bbox"] is not None else None,
            target_center=tuple(row["target_center_px"]) if row["target_center_px"] is not None else None,
            plug_tip=tuple(row["plug_tip_px"]) if row["plug_tip_px"] is not None else None,
            stats=row,
            output_path=overlay_path,
        )

    summary = {
        "annotation_root": str(annotation_root),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "selected_rows": len(selected_rows),
        "slots": slot_counts,
        "criteria": [name for name, _ in criteria],
        "max_per_slot": int(args.max_per_slot),
        "per_criterion": int(args.per_criterion),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
