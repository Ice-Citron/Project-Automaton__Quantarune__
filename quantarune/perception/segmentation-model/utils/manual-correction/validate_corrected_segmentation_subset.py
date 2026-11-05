from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate corrected target/plug masks for the manual segmentation "
            "review subset before retraining."
        )
    )
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--min-target-area-frac", type=float, default=0.0002)
    parser.add_argument("--min-plug-area-frac", type=float, default=0.0001)
    parser.add_argument("--max-components", type=int, default=4)
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


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (mask > 127).astype(np.uint8)


def _mask_stats(mask: np.ndarray) -> tuple[float, int]:
    area_frac = float(mask.mean())
    num_labels, _ = cv2.connectedComponents((mask * 255).astype(np.uint8))
    # connectedComponents counts background too
    component_count = max(0, int(num_labels) - 1)
    return area_frac, component_count


def main() -> int:
    args = parse_args()
    review_root = Path(args.review_root).expanduser().resolve()
    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest is not None
        else review_root / "review_manifest.jsonl"
    )
    rows = _load_jsonl(manifest_path)

    summary: dict[str, Any] = {
        "review_root": str(review_root),
        "manifest": str(manifest_path),
        "rows": len(rows),
        "issues": Counter(),
        "slot_issues": defaultdict(Counter),
    }

    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        rel_image = Path(str(row["image_path"])).relative_to("images")
        target_mask_path = review_root / "masks" / "target" / rel_image
        plug_mask_path = review_root / "masks" / "plug" / rel_image
        slot = str(row.get("target_slot", "unknown"))

        issues: list[str] = []
        try:
            target_mask = _read_mask(target_mask_path)
            plug_mask = _read_mask(plug_mask_path)
        except FileNotFoundError as exc:
            issues.append("missing_mask_file")
            summary["issues"]["missing_mask_file"] += 1
            summary["slot_issues"][slot]["missing_mask_file"] += 1
            audit_rows.append(
                {
                    "image_path": str(row["image_path"]),
                    "target_slot": slot,
                    "issues": issues,
                }
            )
            continue

        target_area_frac, target_components = _mask_stats(target_mask)
        plug_area_frac, plug_components = _mask_stats(plug_mask)

        if target_area_frac < args.min_target_area_frac:
            issues.append("target_mask_too_small")
        if plug_area_frac < args.min_plug_area_frac:
            issues.append("plug_mask_too_small")
        if target_components > args.max_components:
            issues.append("target_mask_too_many_components")
        if plug_components > args.max_components:
            issues.append("plug_mask_too_many_components")

        for issue in issues:
            summary["issues"][issue] += 1
            summary["slot_issues"][slot][issue] += 1

        audit_rows.append(
            {
                "image_path": str(row["image_path"]),
                "target_slot": slot,
                "target_area_frac": target_area_frac,
                "plug_area_frac": plug_area_frac,
                "target_components": target_components,
                "plug_components": plug_components,
                "issues": issues,
            }
        )

    issues_dict = dict(summary["issues"])
    slot_issues_dict = {
        slot: dict(counter) for slot, counter in summary["slot_issues"].items()
    }
    output = {
        "review_root": summary["review_root"],
        "manifest": summary["manifest"],
        "rows": summary["rows"],
        "issues": issues_dict,
        "slot_issues": slot_issues_dict,
        "audit_rows": audit_rows,
    }
    (review_root / "correction_audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: output[k] for k in ("review_root", "manifest", "rows", "issues", "slot_issues")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
