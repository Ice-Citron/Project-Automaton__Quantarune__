from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build corrected-only and mixed corrected+pseudo-label manifests for "
            "segmentation fine-tuning."
        )
    )
    parser.add_argument("--review-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pseudo-train-manifest", default=None)
    parser.add_argument("--pseudo-val-manifest", default=None)
    parser.add_argument("--pseudo-data-root", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reviewed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only rows marked review_done in correction_progress.json.",
    )
    parser.add_argument(
        "--corrected-train-repeat",
        type=int,
        default=4,
        help="Repeat corrected train rows in the mixed train manifest for oversampling.",
    )
    parser.add_argument(
        "--pseudo-train-sample-frac",
        type=float,
        default=1.0,
        help="Fraction of pseudo train rows to keep in the mixed train manifest.",
    )
    parser.add_argument(
        "--pseudo-val-sample-frac",
        type=float,
        default=1.0,
        help="Fraction of pseudo val rows to keep in the mixed val manifest.",
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _absolute_corrected_row(review_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    image_rel = Path(str(row["image_path"]))
    if image_rel.parts and image_rel.parts[0] == "images":
        image_subpath = Path(*image_rel.parts[1:])
    else:
        image_subpath = image_rel

    corrected = dict(row)
    corrected["image_path"] = str((review_root / "images" / image_subpath).resolve())
    corrected["target_mask_path"] = str(
        (review_root / "masks" / "target" / image_subpath).resolve()
    )
    corrected["plug_mask_path"] = str(
        (review_root / "masks" / "plug" / image_subpath).resolve()
    )
    corrected["label_source"] = "corrected_review"
    return corrected


def _absolute_pseudo_row(pseudo_root: Path, row: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(row)
    for key in ("image_path", "target_mask_path", "plug_mask_path"):
        value = resolved.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                resolved[key] = str((pseudo_root / path).resolve())
    resolved["label_source"] = "pseudo_label"
    return resolved


def _sample_rows(rows: list[dict[str, Any]], frac: float, rng: random.Random) -> list[dict[str, Any]]:
    if frac >= 1.0:
        return list(rows)
    if frac <= 0.0 or not rows:
        return []
    keep = max(1, int(round(len(rows) * frac)))
    sampled = list(rows)
    rng.shuffle(sampled)
    return sampled[:keep]


def main() -> int:
    args = parse_args()
    review_root = Path(args.review_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_candidates = (
        review_root / "review_manifest.jsonl",
        review_root / "manifest.jsonl",
    )
    review_manifest_path = next((path for path in manifest_candidates if path.is_file()), None)
    if review_manifest_path is None:
        raise SystemExit(f"No review manifest found under {review_root}")

    progress = _load_progress(review_root / "correction_progress.json")
    review_rows = _load_jsonl(review_manifest_path)

    selected_rows: list[dict[str, Any]] = []
    skipped_unreviewed = 0
    for row in review_rows:
        key = str(row["image_path"])
        reviewed = bool(progress.get(key, {}).get("review_done", False))
        if args.reviewed_only and not reviewed:
            skipped_unreviewed += 1
            continue
        selected_rows.append(_absolute_corrected_row(review_root, row))

    if not selected_rows:
        raise SystemExit(
            "No corrected rows selected. Mark frames reviewed in correction_progress.json "
            "or use --no-reviewed-only."
        )

    rng = random.Random(args.seed)
    episode_ids = sorted({str(row.get("episode_id", "unknown")) for row in selected_rows})
    rng.shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * args.val_fraction)))
    val_episode_ids = set(episode_ids[:val_count])

    corrected_train = [
        row for row in selected_rows if str(row.get("episode_id", "unknown")) not in val_episode_ids
    ]
    corrected_val = [
        row for row in selected_rows if str(row.get("episode_id", "unknown")) in val_episode_ids
    ]

    _write_jsonl(output_root / "corrected_train.jsonl", corrected_train)
    _write_jsonl(output_root / "corrected_val.jsonl", corrected_val)

    summary: dict[str, Any] = {
        "review_root": str(review_root),
        "review_manifest": str(review_manifest_path),
        "reviewed_only": bool(args.reviewed_only),
        "selected_rows": len(selected_rows),
        "skipped_unreviewed": skipped_unreviewed,
        "corrected_train_rows": len(corrected_train),
        "corrected_val_rows": len(corrected_val),
        "corrected_episode_count": len(episode_ids),
        "corrected_val_episode_count": len(val_episode_ids),
    }

    if args.pseudo_train_manifest and args.pseudo_val_manifest:
        if args.pseudo_data_root is None:
            raise SystemExit("--pseudo-data-root is required when pseudo manifests are provided.")

        pseudo_root = Path(args.pseudo_data_root).expanduser().resolve()
        pseudo_train_rows = [
            _absolute_pseudo_row(pseudo_root, row)
            for row in _load_jsonl(Path(args.pseudo_train_manifest).expanduser().resolve())
        ]
        pseudo_val_rows = [
            _absolute_pseudo_row(pseudo_root, row)
            for row in _load_jsonl(Path(args.pseudo_val_manifest).expanduser().resolve())
        ]

        corrected_image_paths = {str(row["image_path"]) for row in selected_rows}
        pseudo_train_rows = [
            row for row in pseudo_train_rows if str(row["image_path"]) not in corrected_image_paths
        ]
        pseudo_val_rows = [
            row for row in pseudo_val_rows if str(row["image_path"]) not in corrected_image_paths
        ]

        pseudo_train_rows = _sample_rows(pseudo_train_rows, args.pseudo_train_sample_frac, rng)
        pseudo_val_rows = _sample_rows(pseudo_val_rows, args.pseudo_val_sample_frac, rng)

        mixed_train = (corrected_train * max(1, int(args.corrected_train_repeat))) + pseudo_train_rows
        mixed_val = corrected_val + pseudo_val_rows
        rng.shuffle(mixed_train)
        rng.shuffle(mixed_val)

        _write_jsonl(output_root / "mixed_train.jsonl", mixed_train)
        _write_jsonl(output_root / "mixed_val.jsonl", mixed_val)

        summary.update(
            {
                "pseudo_data_root": str(pseudo_root),
                "pseudo_train_rows_kept": len(pseudo_train_rows),
                "pseudo_val_rows_kept": len(pseudo_val_rows),
                "corrected_train_repeat": int(args.corrected_train_repeat),
                "mixed_train_rows": len(mixed_train),
                "mixed_val_rows": len(mixed_val),
                "pseudo_train_sample_frac": float(args.pseudo_train_sample_frac),
                "pseudo_val_sample_frac": float(args.pseudo_val_sample_frac),
            }
        )

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
