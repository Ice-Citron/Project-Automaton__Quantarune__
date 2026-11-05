from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train/val manifests for the learned port segmentation dataset "
            "from an annotation pack and aligned mask directories."
        )
    )
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-mask-root", default=None)
    parser.add_argument("--plug-mask-root", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--plug-mask-required",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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


def _relative_mask_path(image_rel: str, mask_root: str) -> Path:
    image_path = Path(image_rel)
    if image_path.parts and image_path.parts[0] == "images":
        image_path = Path(*image_path.parts[1:])
    return Path(mask_root) / image_path


def main() -> int:
    args = parse_args()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    target_mask_root = args.target_mask_root or "masks/target"
    plug_mask_root = args.plug_mask_root or "masks/plug"

    metadata_rows = _load_jsonl(annotation_root / "metadata.jsonl")
    if not metadata_rows:
        raise SystemExit(f"No metadata rows found in {annotation_root / 'metadata.jsonl'}")

    usable_rows: list[dict[str, Any]] = []
    missing_target = 0
    missing_plug = 0
    for row in metadata_rows:
        image_rel = str(row["image_path"])
        target_rel = _relative_mask_path(image_rel, target_mask_root)
        plug_rel = _relative_mask_path(image_rel, plug_mask_root)

        target_abs = annotation_root / target_rel
        plug_abs = annotation_root / plug_rel
        if not target_abs.is_file():
            missing_target += 1
            continue

        manifest_row = dict(row)
        manifest_row["target_mask_path"] = str(target_rel)
        if plug_abs.is_file():
            manifest_row["plug_mask_path"] = str(plug_rel)
        elif args.plug_mask_required:
            missing_plug += 1
            continue
        else:
            manifest_row["plug_mask_path"] = None

        usable_rows.append(manifest_row)

    if not usable_rows:
        raise SystemExit("No usable annotation rows found after mask checks.")

    episode_ids = sorted({str(row.get("episode_id", "unknown")) for row in usable_rows})
    rng = random.Random(args.seed)
    rng.shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * args.val_fraction)))
    val_episode_ids = set(episode_ids[:val_count])

    train_rows = [row for row in usable_rows if str(row.get("episode_id", "unknown")) not in val_episode_ids]
    val_rows = [row for row in usable_rows if str(row.get("episode_id", "unknown")) in val_episode_ids]

    train_path = output_root / "train.jsonl"
    val_path = output_root / "val.jsonl"
    for path, rows in ((train_path, train_rows), (val_path, val_rows)):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "annotation_root": str(annotation_root),
        "output_root": str(output_root),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "episodes_total": len(episode_ids),
        "episodes_val": len(val_episode_ids),
        "missing_target_masks": missing_target,
        "missing_plug_masks": missing_plug,
        "target_mask_root": target_mask_root,
        "plug_mask_root": plug_mask_root,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
