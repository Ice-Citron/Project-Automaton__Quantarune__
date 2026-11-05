from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDNAMES = [
    "stage_dataset_episode_index",
    "stage_name",
    "episode_uid",
    "session_name",
    "experiment_name",
    "case_name",
    "source_type",
    "source_batch_name",
    "source_cases_csv",
    "source_case_name",
    "source_bucket",
    "planned_bucket_label",
    "training_bucket_label",
    "bucket_label",
    "success_for_training",
    "analysis_only_reason",
    "source_split",
    "planned_demo_bucket",
    "expected_primary_stage",
    "recommended_collection_mode",
    "generation_mode",
    "sc_demo_bucket_profile",
    "rollout_profile",
    "perturbation_profile",
    "score_total",
    "score_tier_1",
    "score_tier_2",
    "score_tier_3",
    "insertion_event_fired",
    "approx_max_force_n",
    "force_quality_label",
    "force_penalty_applied",
    "force_above_20n_seen",
    "side_contact_happened",
    "backout_happened",
    "partial_insertion_happened",
    "reseat_happened",
    "dataset_root",
    "target_module_name",
    "task_result_success_last",
    "task_result_message_last",
    "segment",
    "scene_env",
    "segmentation_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize per-stage stage_segment_index.csv files directly from "
            "an SC stage supervision JSON."
        )
    )
    parser.add_argument(
        "--stage-supervision-json",
        required=True,
        help="Path to sc_fsm_stage_supervision.json.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Datasets root under which approach/align/guarded_insert/retry_or_finish live.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing stage index CSVs if present.",
    )
    return parser.parse_args()


def _json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _build_row(episode: dict[str, Any], segment: dict[str, Any], episode_index: int) -> dict[str, Any]:
    return {
        "stage_dataset_episode_index": episode_index,
        "stage_name": segment["stage_name"],
        "episode_uid": episode.get("episode_uid", ""),
        "session_name": episode.get("session_name", ""),
        "experiment_name": episode.get("experiment_name", ""),
        "case_name": episode.get("case_name", ""),
        "source_type": episode.get("source_type", ""),
        "source_batch_name": episode.get("source_batch_name", ""),
        "source_cases_csv": episode.get("source_cases_csv", ""),
        "source_case_name": episode.get("source_case_name", ""),
        "source_bucket": episode.get("source_bucket", ""),
        "planned_bucket_label": episode.get("planned_bucket_label", ""),
        "training_bucket_label": episode.get("training_bucket_label", ""),
        "bucket_label": episode.get("bucket_label", ""),
        "success_for_training": episode.get("success_for_training", ""),
        "analysis_only_reason": episode.get("analysis_only_reason", ""),
        "source_split": episode.get("source_split", ""),
        "planned_demo_bucket": episode.get("planned_demo_bucket", ""),
        "expected_primary_stage": episode.get("expected_primary_stage", ""),
        "recommended_collection_mode": episode.get("recommended_collection_mode", ""),
        "generation_mode": episode.get("generation_mode", ""),
        "sc_demo_bucket_profile": episode.get("sc_demo_bucket_profile", ""),
        "rollout_profile": episode.get("rollout_profile", ""),
        "perturbation_profile": episode.get("perturbation_profile", ""),
        "score_total": episode.get("score_total", ""),
        "score_tier_1": episode.get("score_tier_1", ""),
        "score_tier_2": episode.get("score_tier_2", ""),
        "score_tier_3": episode.get("score_tier_3", ""),
        "insertion_event_fired": episode.get("insertion_event_fired", ""),
        "approx_max_force_n": episode.get("approx_max_force_n", ""),
        "force_quality_label": episode.get("force_quality_label", ""),
        "force_penalty_applied": episode.get("force_penalty_applied", ""),
        "force_above_20n_seen": episode.get("force_above_20n_seen", ""),
        "side_contact_happened": episode.get("side_contact_happened", ""),
        "backout_happened": episode.get("backout_happened", ""),
        "partial_insertion_happened": episode.get("partial_insertion_happened", ""),
        "reseat_happened": episode.get("reseat_happened", ""),
        "dataset_root": episode.get("dataset_root", ""),
        "target_module_name": episode.get("target_module_name", ""),
        "task_result_success_last": episode.get("task_result_success_last", ""),
        "task_result_message_last": episode.get("task_result_message_last", ""),
        "segment": _json_string(segment),
        "scene_env": _json_string(episode.get("scene_env", {})),
        "segmentation_summary": _json_string(episode.get("segmentation", {})),
    }


def main() -> int:
    args = parse_args()
    supervision_path = Path(args.stage_supervision_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    if not supervision_path.is_file():
        raise SystemExit(f"Missing stage supervision JSON: {supervision_path}")

    manifest = json.loads(supervision_path.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    declared_stage_names = list(manifest.get("stage_names", []))
    rows_by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in declared_stage_names}

    for episode in episodes:
        for segment in episode.get("segments", []):
            stage_name = str(segment.get("stage_name", "")).strip()
            if not stage_name:
                continue
            rows = rows_by_stage.setdefault(stage_name, [])
            rows.append(_build_row(episode, segment, len(rows)))

    build_manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "source_supervision_json": str(supervision_path),
        "datasets_root": str(output_root),
        "datasets": {},
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for stage_name, rows in rows_by_stage.items():
        stage_root = output_root / stage_name
        stage_root.mkdir(parents=True, exist_ok=True)
        csv_path = stage_root / "stage_segment_index.csv"
        manifest_path = stage_root / "stage_dataset_manifest.json"
        if (csv_path.exists() or manifest_path.exists()) and not args.force:
            raise SystemExit(
                f"Stage index already exists for {stage_name}: {csv_path}. Re-run with --force."
            )
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        stage_manifest = {
            "created_at": datetime.now().isoformat(),
            "stage_name": stage_name,
            "dataset_root": str(stage_root),
            "episodes_total": len(rows),
            "frames_total": sum(
                int(json.loads(row["segment"]).get("frame_count", 0))
                for row in rows
            ),
            "source_supervision_json": str(supervision_path),
            "segments": rows,
        }
        manifest_path.write_text(
            json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        build_manifest["datasets"][stage_name] = {
            "dataset_root": str(stage_root),
            "episodes_total": len(rows),
            "frames_total": stage_manifest["frames_total"],
        }

    (output_root / "stage_dataset_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Stage index datasets written under: {output_root}")
    for stage_name in declared_stage_names:
        summary = build_manifest["datasets"].get(stage_name, {})
        print(
            f"  {stage_name}: episodes={summary.get('episodes_total', 0)} "
            f"frames={summary.get('frames_total', 0)} "
            f"root={summary.get('dataset_root', '<missing>')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
