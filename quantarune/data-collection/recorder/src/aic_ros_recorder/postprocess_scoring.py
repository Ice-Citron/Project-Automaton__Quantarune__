import argparse
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from lerobot.datasets.utils import load_info, load_stats, write_info, write_stats

SCORE_FEATURES = {
    "score_tier_1": {"dtype": "float32", "shape": (1,), "names": None},
    "score_tier_2": {"dtype": "float32", "shape": (1,), "names": None},
    "score_tier_3": {"dtype": "float32", "shape": (1,), "names": None},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate an AIC recorder dataset with scores from scoring.yaml"
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset root (the directory containing data/, meta/, videos/).",
    )
    parser.add_argument(
        "--scoring-yaml",
        type=Path,
        default=None,
        help="Optional explicit path to scoring.yaml. Defaults to <dataset_root>/scoring.yaml.",
    )
    return parser.parse_args()


def find_scoring_yaml(dataset_root: Path, scoring_yaml: Path | None) -> Path:
    if scoring_yaml is not None:
        return scoring_yaml
    return dataset_root / "scoring.yaml"


def load_trial_scores(scoring_yaml: Path) -> dict[int, dict[str, float]]:
    if not scoring_yaml.exists():
        raise FileNotFoundError(f"scoring.yaml not found: {scoring_yaml}")

    with scoring_yaml.open("r", encoding="utf-8") as file:
        scoring_data = yaml.safe_load(file) or {}

    trial_scores: dict[int, dict[str, float]] = {}
    trial_pattern = re.compile(r"^trial_(\d+)$")
    for key, value in scoring_data.items():
        match = trial_pattern.match(str(key))
        if not match:
            continue

        episode_index = int(match.group(1)) - 1
        trial_scores[episode_index] = {
            "score_tier_1": float(value["tier_1"]["score"]),
            "score_tier_2": float(value["tier_2"]["score"]),
            "score_tier_3": float(value["tier_3"]["score"]),
        }

    if not trial_scores:
        raise ValueError(f"No trial_* entries were found in {scoring_yaml}")

    return trial_scores


def replace_or_append_column(
    table: pa.Table, column_name: str, values: list[float]
) -> pa.Table:
    array = pa.array(values, type=pa.float32())
    field_index = table.schema.get_field_index(column_name)
    if field_index >= 0:
        table = table.remove_column(field_index)
        table = table.add_column(field_index, column_name, array)
    else:
        table = table.append_column(column_name, array)
    return table


def write_table_atomic(table: pa.Table, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(".tmp.parquet")
    pq.write_table(table, tmp_path, compression="snappy", use_dictionary=True)
    tmp_path.replace(output_path)


def annotate_parquet_files(
    parquet_paths: list[Path],
    score_by_episode: dict[int, dict[str, float]],
) -> dict[str, list[np.ndarray]]:
    collected_scores = {name: [] for name in SCORE_FEATURES}

    for parquet_path in sorted(parquet_paths):
        table = pq.read_table(parquet_path)
        episode_indices = [
            int(value.as_py()) for value in table.column("episode_index")
        ]

        score_columns: dict[str, list[float]] = {name: [] for name in SCORE_FEATURES}
        for episode_index in episode_indices:
            if episode_index not in score_by_episode:
                raise ValueError(
                    f"Episode index {episode_index} from {parquet_path} "
                    f"has no corresponding trial in scoring.yaml"
                )
            episode_scores = score_by_episode[episode_index]
            for score_name in SCORE_FEATURES:
                score_columns[score_name].append(float(episode_scores[score_name]))

        for score_name, values in score_columns.items():
            table = replace_or_append_column(table, score_name, values)
            collected_scores[score_name].append(np.asarray(values, dtype=np.float32))

        write_table_atomic(table, parquet_path)

    return collected_scores


def compute_scalar_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "min": np.array([np.nan], dtype=np.float32),
            "max": np.array([np.nan], dtype=np.float32),
            "mean": np.array([np.nan], dtype=np.float32),
            "std": np.array([np.nan], dtype=np.float32),
            "count": np.array([0], dtype=np.int64),
            "q01": np.array([np.nan], dtype=np.float32),
            "q10": np.array([np.nan], dtype=np.float32),
            "q50": np.array([np.nan], dtype=np.float32),
            "q90": np.array([np.nan], dtype=np.float32),
            "q99": np.array([np.nan], dtype=np.float32),
        }

    quantiles = np.quantile(finite_values, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "min": np.array([np.min(finite_values)], dtype=np.float32),
        "max": np.array([np.max(finite_values)], dtype=np.float32),
        "mean": np.array([np.mean(finite_values)], dtype=np.float32),
        "std": np.array([np.std(finite_values)], dtype=np.float32),
        "count": np.array([finite_values.size], dtype=np.int64),
        "q01": np.array([quantiles[0]], dtype=np.float32),
        "q10": np.array([quantiles[1]], dtype=np.float32),
        "q50": np.array([quantiles[2]], dtype=np.float32),
        "q90": np.array([quantiles[3]], dtype=np.float32),
        "q99": np.array([quantiles[4]], dtype=np.float32),
    }


def update_dataset_metadata(
    dataset_root: Path,
    collected_scores: dict[str, list[np.ndarray]],
) -> None:
    info = load_info(dataset_root)
    for score_name, feature in SCORE_FEATURES.items():
        info["features"][score_name] = feature
    write_info(info, dataset_root)

    stats = load_stats(dataset_root) or {}
    for score_name, arrays in collected_scores.items():
        stats[score_name] = compute_scalar_stats(np.concatenate(arrays))
    write_stats(stats, dataset_root)


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    scoring_yaml = find_scoring_yaml(dataset_root, args.scoring_yaml).resolve()

    if not (dataset_root / "data").exists():
        raise FileNotFoundError(f"Dataset root does not contain data/: {dataset_root}")

    trial_scores = load_trial_scores(scoring_yaml)

    data_parquets = list((dataset_root / "data").glob("*/*.parquet"))
    if not data_parquets:
        raise FileNotFoundError(
            f"No data parquet files were found under {dataset_root / 'data'}"
        )

    episodes_parquets = list((dataset_root / "meta" / "episodes").glob("*/*.parquet"))

    collected_scores = annotate_parquet_files(data_parquets, trial_scores)
    if episodes_parquets:
        annotate_parquet_files(episodes_parquets, trial_scores)

    update_dataset_metadata(dataset_root, collected_scores)

    print(f"Annotated dataset at {dataset_root} using scores from {scoring_yaml}")
    print(
        "Added columns: "
        + ", ".join(SCORE_FEATURES)
        + " to data parquet files"
        + (" and meta/episodes parquet files." if episodes_parquets else ".")
    )


if __name__ == "__main__":
    main()
