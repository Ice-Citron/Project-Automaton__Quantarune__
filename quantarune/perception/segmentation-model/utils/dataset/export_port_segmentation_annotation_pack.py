from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
RUNTIME_POLICY_ROOT = WORKSPACE_ROOT / "aic-quantarune" / "aic_example_policies"
if str(RUNTIME_POLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_POLICY_ROOT))

from quantarune_training.lerobot_video_patch import patch_lerobot_video_backend
from quantarune_training.vision.schema import TASK_CONDITION_NAMES
from aic_example_policies.ros.vision_target_features import build_task_grounding_metadata

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export center-camera ACT frames plus task metadata into an annotation "
            "pack for learned port segmentation."
        )
    )
    parser.add_argument("--source", required=True, help="Input ACT-ready dataset root.")
    parser.add_argument("--output", required=True, help="Output annotation-pack root.")
    parser.add_argument("--camera", default="center_camera", help="Camera key to export.")
    parser.add_argument("--stride", type=int, default=10, help="Sample every Nth frame.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite output when it already exists.",
    )
    return parser.parse_args()


def _load_task_text_lookup(dataset_root: Path) -> dict[int, str]:
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    tasks_df = pd.read_parquet(tasks_path)
    text_column = next(
        (column for column in tasks_df.columns if column != "task_index"),
        None,
    )
    lookup: dict[int, str] = {}
    for row in tasks_df.to_dict(orient="records"):
        task_index = int(row["task_index"])
        lookup[task_index] = str(row.get(text_column, task_index))
    return lookup


def _state_feature_names(feature_meta: dict[str, Any]) -> list[str]:
    names = feature_meta.get("names")
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return list(names)
    shape = tuple(feature_meta.get("shape", ()))
    if len(shape) != 1:
        raise SystemExit(
            "Expected observation.state to be a flat vector. "
            f"Received shape={shape!r}."
        )
    return [f"state_{index:03d}" for index in range(shape[0])]


def _derive_episode_ranges(episode_indices: list[int]) -> list[tuple[int, int, int]]:
    if not episode_indices:
        return []

    ranges: list[tuple[int, int, int]] = []
    start = 0
    current_episode = int(episode_indices[0])
    for idx, episode_index in enumerate(episode_indices[1:], start=1):
        if int(episode_index) == current_episode:
            continue
        ranges.append((current_episode, start, idx))
        start = idx
        current_episode = int(episode_index)
    ranges.append((current_episode, start, len(episode_indices)))
    return ranges


def _sample_to_rgb(sample_value: Any) -> np.ndarray:
    if hasattr(sample_value, "detach"):
        arr = sample_value.detach().cpu().numpy()
    elif hasattr(sample_value, "cpu"):
        arr = sample_value.cpu().numpy()
    else:
        arr = np.asarray(sample_value)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(np.max(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0.0, 255.0)
        arr = arr.astype(np.uint8)

    return arr[:, :, :3]


def _feature_value(sample_value: Any) -> Any:
    if hasattr(sample_value, "detach"):
        return sample_value.detach().cpu().numpy()
    if hasattr(sample_value, "cpu"):
        return sample_value.cpu().numpy()
    return sample_value


def _scalar_int(sample_value: Any, default: int = -1) -> int:
    value = _feature_value(sample_value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.reshape(-1)[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _task_hint_from_state(
    state_vector: np.ndarray,
    state_feature_names: list[str],
) -> dict[str, float]:
    state_dict = dict(zip(state_feature_names, state_vector.tolist(), strict=False))
    hint = {
        name: float(state_dict.get(name, 0.0))
        for name in TASK_CONDITION_NAMES
    }
    hint["task.target_module_index"] = float(state_dict.get("task.target_module_index", 0.0))
    hint["task.target_port_index"] = float(state_dict.get("task.target_port_index", 0.0))
    return hint


def main() -> int:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Source dataset not found: {source_root}")
    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise SystemExit(
                f"Output annotation pack already exists: {output_root}. Re-run with --force."
            )

    patch_lerobot_video_backend()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=source_root.name,
        root=source_root,
        video_backend="pyav",
    )
    source_features = dataset.meta.features
    state_names = _state_feature_names(source_features["observation.state"])
    task_text_lookup = _load_task_text_lookup(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.jsonl"

    episode_indices = [
        int(_scalar_int(dataset.hf_dataset[idx]["episode_index"], default=-1))
        for idx in range(len(dataset.hf_dataset))
    ]
    episode_ranges = _derive_episode_ranges(episode_indices)
    if args.max_episodes is not None:
        episode_ranges = episode_ranges[: args.max_episodes]

    row_count = 0
    records: list[dict[str, Any]] = []
    iterator = tqdm(episode_ranges, desc="Export episodes") if tqdm else episode_ranges
    for episode_index, start_idx, end_idx in iterator:
        episode_id = f"ep_{episode_index:06d}"
        episode_dir = images_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        local_frame_count = 0
        for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
            if args.max_frames_per_episode is not None and local_frame_count >= args.max_frames_per_episode:
                break
            if local_idx % args.stride != 0:
                continue

            sample = dataset[global_idx]
            state_vector = np.asarray(_feature_value(sample["observation.state"]), dtype=np.float32)
            task_index = _scalar_int(sample.get("task_index"), default=-1)
            task_text = task_text_lookup.get(task_index)
            task_hint = _task_hint_from_state(state_vector, state_names)
            grounding = build_task_grounding_metadata(
                task_hint=task_hint,
                task_text=task_text,
                episode_id=episode_id,
                frame_idx=local_idx,
            )

            image_rgb = _sample_to_rgb(sample[f"observation.images.{args.camera}"])
            image_rel = Path("images") / episode_id / f"frame_{local_idx:06d}.png"
            image_out = output_root / image_rel
            image_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_out), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

            record = {
                **grounding,
                **{name: float(task_hint.get(name, 0.0)) for name in TASK_CONDITION_NAMES},
                "camera": args.camera,
                "image_path": str(image_rel),
                "source_dataset_root": str(source_root),
                "source_global_index": int(global_idx),
                "source_episode_index": int(episode_index),
                "source_task_index": int(task_index),
            }
            records.append(record)
            row_count += 1
            local_frame_count += 1

    with metadata_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "camera": args.camera,
        "stride": int(args.stride),
        "episodes_exported": len({record["episode_id"] for record in records}),
        "frames_exported": row_count,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
