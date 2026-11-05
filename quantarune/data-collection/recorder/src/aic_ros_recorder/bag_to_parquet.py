import argparse
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert recorded rosbag folder to lerobot Parquet format"
    )
    parser.add_argument(
        "--bag-folder",
        type=str,
        required=True,
        help="Path to rosbag output folder (recorded by ros2 bag)",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Target folder for the resulting dataset",
    )
    args = parser.parse_args()

    bag_folder = Path(args.bag_folder)
    output_folder = Path(
        args.output_folder or bag_folder.with_name(bag_folder.name + "_parquet")
    )

    # NOTE: The recorder now writes a LeRobot dataset directly while recording.
    # This script remains as a compatibility placeholder for older rosbag-only captures.

    source_dataset = bag_folder / "episode.parquet"
    if source_dataset.exists():
        output_folder.mkdir(parents=True, exist_ok=True)
        target_parquet = output_folder / "episode.parquet"
        os.replace(str(source_dataset), str(target_parquet))
        print(f"Copied {source_dataset} => {target_parquet}")
        return

    print("No conversion was performed.")
    print(
        "AICRosRecorder now writes the LeRobot dataset directly into experiment_<N>/data, meta and videos."
    )
    print(
        "This script only supports legacy bag folders that already contain a converted episode.parquet."
    )


if __name__ == "__main__":
    main()
