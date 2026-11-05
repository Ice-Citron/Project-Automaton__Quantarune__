from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "quantarune_training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from quantarune_training.lerobot_video_patch import patch_lerobot_video_backend


DEFAULT_CAMERA_KEYS = (
    "observation.images.center_camera",
    "observation.images.left_camera",
    "observation.images.right_camera",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build four stage-specific ACT48-compatible local LeRobot datasets "
            "from the SC teleop stage supervision manifest."
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
        help="Parent directory where stage dataset roots will be created.",
    )
    parser.add_argument(
        "--camera-key",
        action="append",
        dest="camera_keys",
        default=None,
        help="Camera key to keep. Repeat to keep multiple. Defaults to all three RGB cameras.",
    )
    parser.add_argument(
        "--stage-name",
        action="append",
        dest="stage_names",
        default=None,
        help="Only build these stages. Repeat to build multiple.",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=8,
        help="Skip segments shorter than this many frames. Default: 8.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=0.25,
        help="Scale factor applied to RGB camera frames before writing. Default: 0.25.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=10,
        help="Async image-writer thread count. Default: 10.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Async image-writer process count. Default: 0.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help="Episodes to batch before video encoding. Default: 1.",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        choices=("h264", "hevc", "libsvtav1"),
        help="Video codec for encoded LeRobot videos. Default: h264.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing stage dataset roots. Default: false.",
    )
    return parser.parse_args()


def _scaled_image_shape(shape: tuple[int, ...], image_scale: float) -> tuple[int, ...]:
    if len(shape) != 3:
        raise SystemExit(f"Expected camera feature shape to be 3D, received: {shape}")
    height, width, channels = shape
    return (
        max(1, int(round(height * image_scale))),
        max(1, int(round(width * image_scale))),
        channels,
    )


def _resize_frame(image: np.ndarray, image_scale: float) -> np.ndarray:
    if image_scale == 1.0:
        return image
    pil_image = Image.fromarray(image)
    scaled = pil_image.resize(
        (
            max(1, int(round(pil_image.width * image_scale))),
            max(1, int(round(pil_image.height * image_scale))),
        ),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(scaled, dtype=image.dtype)


def _build_target_features(source_features: dict[str, Any], camera_keys: tuple[str, ...], image_scale: float) -> dict[str, Any]:
    target_features: dict[str, Any] = {}
    for camera_key in camera_keys:
        target_features[camera_key] = {
            "dtype": source_features[camera_key]["dtype"],
            "shape": _scaled_image_shape(tuple(source_features[camera_key]["shape"]), image_scale),
            "names": source_features[camera_key].get("names"),
        }
    target_features["observation.state"] = {
        "dtype": source_features["observation.state"]["dtype"],
        "shape": tuple(source_features["observation.state"]["shape"]),
        "names": source_features["observation.state"].get("names"),
    }
    target_features["action"] = {
        "dtype": source_features["action"]["dtype"],
        "shape": tuple(source_features["action"]["shape"]),
        "names": source_features["action"].get("names"),
    }
    return target_features


def _source_features_match(existing_features: dict[str, Any], reference_features: dict[str, Any], camera_keys: tuple[str, ...]) -> bool:
    for camera_key in camera_keys:
        if camera_key not in existing_features or camera_key not in reference_features:
            return False
        if existing_features[camera_key]["dtype"] != reference_features[camera_key]["dtype"]:
            return False
        if tuple(existing_features[camera_key]["shape"]) != tuple(reference_features[camera_key]["shape"]):
            return False

    for key in ("observation.state", "action"):
        if key not in existing_features or key not in reference_features:
            return False
        if existing_features[key]["dtype"] != reference_features[key]["dtype"]:
            return False
        if tuple(existing_features[key]["shape"]) != tuple(reference_features[key]["shape"]):
            return False
        if existing_features[key].get("names") != reference_features[key].get("names"):
            return False

    return True


def _sorted_stage_segments(manifest: dict[str, Any], requested_stage_names: set[str], min_segment_frames: int) -> dict[str, list[dict[str, Any]]]:
    segments_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in manifest.get("episodes", []):
        for segment in episode.get("segments", []):
            stage_name = str(segment.get("stage_name") or "")
            frame_count = int(segment.get("frame_count") or 0)
            if stage_name not in requested_stage_names or frame_count < min_segment_frames:
                continue
            enriched = {
                "episode_uid": episode.get("episode_uid", ""),
                "session_name": episode.get("session_name", ""),
                "experiment_name": episode.get("experiment_name", ""),
                "dataset_root": episode.get("dataset_root", ""),
                "source_type": episode.get("source_type", ""),
                "case_name": episode.get("case_name", ""),
                "source_batch_name": episode.get("source_batch_name", ""),
                "source_cases_csv": episode.get("source_cases_csv", ""),
                "source_case_name": episode.get("source_case_name", ""),
                "source_bucket": episode.get("source_bucket", ""),
                "planned_demo_bucket": episode.get("planned_demo_bucket", ""),
                "expected_primary_stage": episode.get("expected_primary_stage", ""),
                "recommended_collection_mode": episode.get("recommended_collection_mode", ""),
                "rollout_profile": episode.get("rollout_profile", ""),
                "perturbation_profile": episode.get("perturbation_profile", ""),
                "score_total": episode.get("score_total", ""),
                "score_tier_1": episode.get("score_tier_1", ""),
                "score_tier_2": episode.get("score_tier_2", ""),
                "score_tier_3": episode.get("score_tier_3", ""),
                "target_module_name": episode.get("target_module_name", ""),
                "task_result_success_last": episode.get("task_result_success_last", ""),
                "task_result_message_last": episode.get("task_result_message_last", ""),
                "scene_env": episode.get("scene_env", {}),
                "segmentation_summary": episode.get("segmentation", {}),
                "segment": segment,
            }
            segments_by_stage[stage_name].append(enriched)
    return segments_by_stage


def main() -> int:
    args = parse_args()
    supervision_path = Path(args.stage_supervision_json).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    camera_keys = tuple(args.camera_keys or DEFAULT_CAMERA_KEYS)

    if not supervision_path.is_file():
        raise SystemExit(f"stage supervision JSON not found: {supervision_path}")

    manifest = json.loads(supervision_path.read_text())
    declared_stage_names = tuple(manifest.get("stage_names") or ())
    if not declared_stage_names:
        raise SystemExit(f"stage supervision JSON is missing stage_names: {supervision_path}")
    requested_stage_names = set(args.stage_names or declared_stage_names)

    segments_by_stage = _sorted_stage_segments(
        manifest,
        requested_stage_names=requested_stage_names,
        min_segment_frames=args.min_segment_frames,
    )
    if not any(segments_by_stage.values()):
        raise SystemExit("No stage segments matched the requested filters.")

    first_segment = next(segment for segments in segments_by_stage.values() for segment in segments)
    first_source_root = Path(first_segment["dataset_root"]).expanduser().resolve()

    patch_lerobot_video_backend()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source_cache: dict[Path, Any] = {}

    def load_source(root: Path):
        cached = source_cache.get(root)
        if cached is not None:
            return cached
        dataset = LeRobotDataset(
            repo_id=f"local/{root.parents[1].name}_{root.name}",
            root=root,
            video_backend="pyav",
        )
        source_cache[root] = dataset
        return dataset

    first_source = load_source(first_source_root)
    for camera_key in camera_keys:
        if camera_key not in first_source.meta.features:
            raise SystemExit(f"Camera key missing in first source dataset: {camera_key}")

    target_features = _build_target_features(
        first_source.meta.features,
        camera_keys,
        args.image_scale,
    )

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    build_summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "schema": manifest.get("schema", ""),
        "stage_names": list(declared_stage_names),
        "source_supervision_json": str(supervision_path),
        "min_segment_frames": args.min_segment_frames,
        "image_scale": args.image_scale,
        "vcodec": args.vcodec,
        "datasets": {},
    }

    for stage_name in declared_stage_names:
        if stage_name not in requested_stage_names:
            continue
        stage_segments = segments_by_stage.get(stage_name, [])
        if not stage_segments:
            build_summary["datasets"][stage_name] = {
                "dataset_root": "",
                "episodes_total": 0,
                "frames_total": 0,
                "skipped": "no_segments",
            }
            continue

        stage_root = output_root / stage_name
        if stage_root.exists():
            if args.force:
                shutil.rmtree(stage_root)
            else:
                raise SystemExit(
                    f"Stage dataset root already exists: {stage_root}. Re-run with --force."
                )

        target = LeRobotDataset.create(
            repo_id=stage_root.name,
            root=stage_root,
            fps=first_source.meta.fps,
            robot_type=first_source.meta.robot_type,
            features=target_features,
            use_videos=True,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,
            vcodec=args.vcodec,
        )

        segment_rows: list[dict[str, Any]] = []
        episodes_total = 0
        frames_total = 0

        for segment_record in stage_segments:
            source_root = Path(segment_record["dataset_root"]).expanduser().resolve()
            source = load_source(source_root)
            if not _source_features_match(
                source.meta.features,
                first_source.meta.features,
                camera_keys,
            ):
                raise SystemExit(f"Source dataset has incompatible features: {source_root}")

            segment = segment_record["segment"]
            start = int(segment["dataset_frame_start_index"])
            end = int(segment["dataset_frame_end_index"])
            if end <= start:
                continue

            for frame_idx in range(start, end):
                sample = source[frame_idx]
                frame = {
                    "observation.state": sample["observation.state"].cpu().numpy(),
                    "action": sample["action"].cpu().numpy(),
                    "task": f"sc_stage_{stage_name}",
                }
                for camera_key in camera_keys:
                    camera_frame = sample[camera_key].permute(1, 2, 0).contiguous().cpu().numpy()
                    frame[camera_key] = _resize_frame(camera_frame, args.image_scale)
                target.add_frame(frame)
                frames_total += 1

            target.save_episode()
            segment_rows.append(
                {
                    "stage_dataset_episode_index": episodes_total,
                    "stage_name": stage_name,
                    "episode_uid": segment_record["episode_uid"],
                    "session_name": segment_record["session_name"],
                    "experiment_name": segment_record["experiment_name"],
                    "case_name": segment_record["case_name"],
                    "source_type": segment_record["source_type"],
                    "source_batch_name": segment_record["source_batch_name"],
                    "source_cases_csv": segment_record["source_cases_csv"],
                    "source_case_name": segment_record["source_case_name"],
                    "source_bucket": segment_record["source_bucket"],
                    "planned_demo_bucket": segment_record["planned_demo_bucket"],
                    "expected_primary_stage": segment_record["expected_primary_stage"],
                    "recommended_collection_mode": segment_record["recommended_collection_mode"],
                    "rollout_profile": segment_record["rollout_profile"],
                    "perturbation_profile": segment_record["perturbation_profile"],
                    "score_total": segment_record["score_total"],
                    "score_tier_1": segment_record["score_tier_1"],
                    "score_tier_2": segment_record["score_tier_2"],
                    "score_tier_3": segment_record["score_tier_3"],
                    "dataset_root": str(source_root),
                    "target_module_name": segment_record["target_module_name"],
                    "task_result_success_last": segment_record["task_result_success_last"],
                    "task_result_message_last": segment_record["task_result_message_last"],
                    "segment": segment,
                    "scene_env": segment_record["scene_env"],
                    "segmentation_summary": segment_record["segmentation_summary"],
                }
            )
            episodes_total += 1

        stage_manifest = {
            "created_at": datetime.now().isoformat(),
            "stage_name": stage_name,
            "dataset_root": str(stage_root),
            "episodes_total": episodes_total,
            "frames_total": frames_total,
            "source_supervision_json": str(supervision_path),
            "camera_keys": list(camera_keys),
            "image_scale": args.image_scale,
            "segments": segment_rows,
        }
        (stage_root / "stage_dataset_manifest.json").write_text(
            json.dumps(stage_manifest, indent=2, sort_keys=True) + "\n"
        )
        with (stage_root / "stage_segment_index.csv").open("w", newline="") as fh:
            fieldnames = [
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
                "planned_demo_bucket",
                "expected_primary_stage",
                "recommended_collection_mode",
                "rollout_profile",
                "perturbation_profile",
                "score_total",
                "score_tier_1",
                "score_tier_2",
                "score_tier_3",
                "dataset_root",
                "target_module_name",
                "task_result_success_last",
                "task_result_message_last",
                "segment",
                "scene_env",
                "segmentation_summary",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in segment_rows:
                csv_row = dict(row)
                csv_row["segment"] = json.dumps(csv_row["segment"], sort_keys=True)
                csv_row["scene_env"] = json.dumps(csv_row["scene_env"], sort_keys=True)
                csv_row["segmentation_summary"] = json.dumps(
                    csv_row["segmentation_summary"],
                    sort_keys=True,
                )
                writer.writerow(csv_row)

        build_summary["datasets"][stage_name] = {
            "dataset_root": str(stage_root),
            "episodes_total": episodes_total,
            "frames_total": frames_total,
        }

    (output_root / "stage_dataset_build_manifest.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True) + "\n"
    )

    print(f"SC stage datasets written under: {output_root}")
    for stage_name, summary in build_summary["datasets"].items():
        print(
            f"  {stage_name}: episodes={summary.get('episodes_total', 0)} "
            f"frames={summary.get('frames_total', 0)} "
            f"root={summary.get('dataset_root', '<skipped>')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
