import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

# Keep the Hugging Face datasets cache writable before importing LeRobotDataset.
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_datasets_cache"

from aic_ros_recorder.aic_metadata import (
    OBSERVATION_FEATURES,
    task_text_dataset_features,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.constants import ACTION, OBS_STR

ACTION_FEATURES = {
    "cartesian.linear.x": float,
    "cartesian.linear.y": float,
    "cartesian.linear.z": float,
    "cartesian.angular.x": float,
    "cartesian.angular.y": float,
    "cartesian.angular.z": float,
    "cartesian.pose.position.x": float,
    "cartesian.pose.position.y": float,
    "cartesian.pose.position.z": float,
    "cartesian.pose.orientation.x": float,
    "cartesian.pose.orientation.y": float,
    "cartesian.pose.orientation.z": float,
    "cartesian.pose.orientation.w": float,
    "joint.shoulder_pan_joint": float,
    "joint.shoulder_lift_joint": float,
    "joint.elbow_joint": float,
    "joint.wrist_1_joint": float,
    "joint.wrist_2_joint": float,
    "joint.wrist_3_joint": float,
}


def build_default_features() -> dict[str, Any]:
    return {
        **hw_to_dataset_features(OBSERVATION_FEATURES, prefix=OBS_STR, use_video=True),
        **task_text_dataset_features(OBS_STR),
        **hw_to_dataset_features(ACTION_FEATURES, prefix=ACTION, use_video=True),
        "task_result.success": {"dtype": "int64", "shape": (1,), "names": None},
        "task_result.message": {"dtype": "string", "shape": (1,), "names": None},
        "score_tier_1": {"dtype": "float32", "shape": (1,), "names": None},
        "score_tier_2": {"dtype": "float32", "shape": (1,), "names": None},
        "score_tier_3": {"dtype": "float32", "shape": (1,), "names": None},
    }


DEFAULT_FEATURES = build_default_features()


def main():
    args = parse_args()

    if args.use_lerobot_dataset:
        read_lerobot_dataset(args)
        return

    read_parquet_file(args)


def read_lerobot_dataset(args):
    dataset_kwargs = {}
    if args.dataset_root:
        dataset_kwargs["root"] = args.dataset_root

    ds = LeRobotDataset(args.dataset_repo_id, **dataset_kwargs)
    features = ds.features or DEFAULT_FEATURES

    print("Dataset root:", ds.root)
    print("Dataset repo id:", args.dataset_repo_id)
    print("Number of rows:", len(ds.hf_dataset))
    print_feature_summary(features)

    task_lookup, task_columns = load_task_lookup_from_dataset_root(Path(ds.root))
    print_task_summary(task_columns)

    start_index = max(0, len(ds.hf_dataset) - 5)
    print(f"\nLast {len(ds.hf_dataset) - start_index} dataset rows:")
    for row_index in range(start_index, len(ds.hf_dataset)):
        raw_row = ds.hf_dataset[row_index]
        expanded_row = expand_record(raw_row, features, task_lookup)
        print(f"\nRow {row_index}:")
        print(json.dumps(expanded_row, indent=2, sort_keys=False))


def read_parquet_file(args):
    parquet_path = Path(args.parquet_file)
    df = pd.read_parquet(parquet_path)
    features = load_parquet_features(parquet_path)
    task_lookup, task_columns = load_task_lookup_from_dataset_root(
        find_dataset_root(parquet_path)
    )

    print("Parquet path:", parquet_path)
    print("Number of rows:", len(df))
    print("Columns:", df.columns.tolist())
    print_feature_summary(features)
    print_task_summary(task_columns)

    start_index = max(0, len(df) - 5)
    print(f"\nLast {len(df) - start_index} rows from the parquet file:")
    for row_index, row in enumerate(
        df.tail(5).to_dict(orient="records"), start=start_index
    ):
        expanded_row = expand_record(row, features, task_lookup)
        print(f"\nRow {row_index}:")
        print(json.dumps(expanded_row, indent=2, sort_keys=False))


def print_feature_summary(features: dict[str, Any]):
    if not features:
        print("Feature schema: not available.")
        return

    print("Feature schema:")
    for feature_name, info in features.items():
        dtype = info.get("dtype", "unknown")
        shape = info.get("shape")
        names = info.get("names")
        summary = f"- {feature_name}: dtype={dtype}, shape={shape}"
        if names:
            summary += f", named values={names}"
        print(summary)


def print_task_summary(task_columns: list[str]):
    if not task_columns:
        print("Command/task text: no metadata file was found.")
        return

    if task_columns == ["task_index"]:
        print(
            "Command/task text: not available in metadata, only task_index is stored."
        )
        return

    print("Command/task metadata columns:", task_columns)


def find_dataset_root(parquet_path: Path) -> Path | None:
    for parent in [parquet_path.parent, *parquet_path.parents]:
        if (parent / "meta" / "info.json").exists():
            return parent
    return None


def load_parquet_features(parquet_path: Path) -> dict[str, Any]:
    dataset_root = find_dataset_root(parquet_path)
    if dataset_root is None:
        return DEFAULT_FEATURES

    info_path = dataset_root / "meta" / "info.json"
    try:
        with info_path.open("r", encoding="utf-8") as file:
            return json.load(file).get("features", DEFAULT_FEATURES)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_FEATURES


def load_task_lookup_from_dataset_root(
    dataset_root: Path | None,
) -> tuple[dict[int, Any], list[str]]:
    if dataset_root is None:
        return {}, []

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}, []

    tasks_df = pd.read_parquet(tasks_path)
    task_columns = tasks_df.columns.tolist()

    if "task_index" not in tasks_df.columns:
        return {}, task_columns

    value_columns = [column for column in task_columns if column != "task_index"]
    if not value_columns:
        return {}, task_columns

    task_lookup: dict[int, Any] = {}
    for record in tasks_df.to_dict(orient="records"):
        task_index = record.get("task_index")
        if task_index is None:
            continue

        sanitized_values = {
            column: to_builtin(record[column]) for column in value_columns
        }
        if len(value_columns) == 1:
            task_lookup[int(task_index)] = sanitized_values[value_columns[0]]
        else:
            task_lookup[int(task_index)] = sanitized_values

    return task_lookup, task_columns


def expand_record(
    record: dict[str, Any], features: dict[str, Any], task_lookup: dict[int, Any]
) -> dict[str, Any]:
    expanded_record: dict[str, Any] = {}

    for key, value in record.items():
        feature_info = features.get(key, {})
        expanded_record[key] = expand_feature_value(value, feature_info)

    task_index = expanded_record.get("task_index")
    if task_index is not None:
        resolved_task = task_lookup.get(int(task_index))
        if resolved_task is not None:
            if isinstance(resolved_task, dict):
                expanded_record["task_metadata"] = resolved_task
                if "task" in resolved_task and "task" not in expanded_record:
                    expanded_record["task"] = resolved_task["task"]
                if "command" in resolved_task and "command" not in expanded_record:
                    expanded_record["command"] = resolved_task["command"]
            elif "command" not in expanded_record:
                expanded_record["command"] = resolved_task

    if "command" not in expanded_record:
        observation_command = expanded_record.get("observation.task.command")
        if observation_command not in (None, ""):
            expanded_record["command"] = observation_command

    if "task" not in expanded_record and "command" in expanded_record:
        expanded_record["task"] = expanded_record["command"]

    return expanded_record


def expand_feature_value(value: Any, feature_info: dict[str, Any]) -> Any:
    builtin_value = to_builtin(value)
    feature_names = feature_info.get("names")

    if (
        isinstance(builtin_value, list)
        and feature_names
        and len(feature_names) == len(builtin_value)
    ):
        return {
            feature_name: to_builtin(feature_value)
            for feature_name, feature_value in zip(feature_names, builtin_value)
        }

    return builtin_value


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(sub_value) for key, sub_value in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return to_builtin(value.tolist())
        except TypeError:
            pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a LeRobot dataset or a parquet file and print all stored rows."
    )
    parser.add_argument(
        "--parquet_file",
        type=str,
        default="file-000.parquet",
        help="Path to the parquet file to inspect in pandas mode.",
    )
    parser.add_argument(
        "--use_lerobot_dataset",
        action="store_true",
        help="Use LeRobotDataset instead of reading a parquet file directly.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="shng2025/record-test-intrinsic-v4",
        help="Repository id used to open the LeRobot dataset.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional local dataset root passed to LeRobotDataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
