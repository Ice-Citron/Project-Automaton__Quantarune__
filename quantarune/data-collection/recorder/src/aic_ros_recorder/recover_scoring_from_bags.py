import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BAG_DIR_RE = re.compile(r"^bag_(trial_\d+)_(\d{8}_\d{6}_\d{3})$")


@dataclass(frozen=True)
class Connection:
    cable_name: str
    task_board_name: str
    plug_name: str
    target_module_name: str
    port_name: str

    def plug_tf_name(self) -> str:
        return f"{self.cable_name}/{self.plug_name}_link"

    def port_tf_name(self) -> str:
        return f"{self.task_board_name}/{self.target_module_name}/{self.port_name}_link"

    def port_entrance_tf_name(self) -> str:
        return f"{self.task_board_name}/{self.target_module_name}/{self.port_name}_link_entrance"


@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class TransformRecord:
    stamp_ns: int
    x: float
    y: float
    z: float


@dataclass
class CategoryScore:
    score: float
    message: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover a scoring.yaml from completed AIC scoring bags. "
            "If the host environment cannot read mcap bags, the script will "
            "re-run itself inside the aic_eval distrobox."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.home() / "aic_results",
        help="Directory containing bag_trial_* folders. Defaults to ~/aic_results.",
    )
    parser.add_argument(
        "--scenario-yaml",
        type=Path,
        required=True,
        help="Scenario/config YAML used for the run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output scoring.yaml path. Defaults to <results-dir>/scoring.yaml.",
    )
    parser.add_argument(
        "--container-name",
        default="aic_eval",
        help="Distrobox container name used for fallback execution. Defaults to aic_eval.",
    )
    parser.add_argument(
        "--gripper-frame",
        default="gripper/tcp",
        help="Gripper frame name used by scoring. Defaults to gripper/tcp.",
    )
    parser.add_argument(
        "--trials",
        nargs="*",
        default=None,
        help="Optional subset of trial ids to recover, e.g. trial_1 trial_2.",
    )
    parser.add_argument(
        "--task-board-name",
        default="task_board",
        help="Task board entity name used in TF namespaces. Defaults to task_board.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Fail instead of overwriting the output scoring.yaml if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra recovery details.",
    )
    parser.add_argument(
        "--_run-inside-container",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def stringify_argv(args: argparse.Namespace, include_inside_flag: bool) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--results-dir",
        str(args.results_dir),
        "--scenario-yaml",
        str(args.scenario_yaml),
        "--container-name",
        args.container_name,
        "--gripper-frame",
        args.gripper_frame,
        "--task-board-name",
        args.task_board_name,
    ]
    if args.output is not None:
        argv.extend(["--output", str(args.output)])
    if args.trials:
        argv.extend(["--trials", *args.trials])
    if args.keep_existing_output:
        argv.append("--keep-existing-output")
    if args.verbose:
        argv.append("--verbose")
    if include_inside_flag:
        argv.append("--_run-inside-container")
    return argv


def rerun_inside_container(args: argparse.Namespace) -> int:
    inner_argv = stringify_argv(args, include_inside_flag=True)
    inner_argv[0] = "python3"
    shell_cmd = (
        "source /ws_aic/install/setup.bash && "
        "export AIC_RECOVER_SCORING_IN_CONTAINER=1 && " + shlex.join(inner_argv)
    )
    cmd = [
        "distrobox",
        "enter",
        args.container_name,
        "--",
        "bash",
        "-lc",
        shell_cmd,
    ]
    return subprocess.run(cmd, check=False).returncode


def metadata_path_for_bag(bag_dir: Path) -> Path:
    return bag_dir / "metadata.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def latest_complete_bags(
    results_dir: Path, selected_trials: set[str] | None
) -> tuple[dict[str, Path], list[Path]]:
    latest: dict[str, tuple[str, Path]] = {}
    skipped_incomplete: list[Path] = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        match = BAG_DIR_RE.match(entry.name)
        if not match:
            continue
        trial_id, stamp = match.groups()
        if selected_trials is not None and trial_id not in selected_trials:
            continue
        if not metadata_path_for_bag(entry).exists():
            skipped_incomplete.append(entry)
            continue
        current = latest.get(trial_id)
        if current is None or stamp > current[0]:
            latest[trial_id] = (stamp, entry)
    return {
        trial_id: path for trial_id, (_, path) in latest.items()
    }, skipped_incomplete


def score_total(categories: dict[str, CategoryScore]) -> float:
    return sum(category.score for category in categories.values())


def category_to_yaml(category: CategoryScore) -> dict[str, Any]:
    node: dict[str, Any] = {"score": float(category.score)}
    if category.message is not None:
        node["message"] = category.message
    return node


def tier_to_yaml(
    score: float, message: str, categories: dict[str, CategoryScore] | None = None
) -> dict[str, Any]:
    node: dict[str, Any] = {"score": float(score), "message": message}
    if categories:
        node["categories"] = {
            name: category_to_yaml(category) for name, category in categories.items()
        }
    return node


def fixed2(value: float) -> str:
    return f"{value:.2f}"


def inverse_proportional_score(
    max_score: float,
    min_score: float,
    max_range: float,
    min_range: float,
    measurement: float,
) -> float:
    if measurement >= max_range:
        return min_score
    if measurement <= min_range:
        return max_score
    return min_score + ((max_range - measurement) / (max_range - min_range)) * (
        max_score - min_score
    )


def bag_time_to_seconds(stamp_ns: int) -> float:
    return float(stamp_ns) * 1e-9


def lookup_transform(
    buffer, target_frame: str, source_frame: str, stamp_ns: int | None
):
    import rclpy
    from tf2_ros import TransformException

    query_time = (
        rclpy.time.Time() if stamp_ns is None else rclpy.time.Time(nanoseconds=stamp_ns)
    )
    try:
        return buffer.lookup_transform(target_frame, source_frame, query_time.to_msg())
    except TransformException:
        return None


def lookup_transform_with_fallback(
    buffer,
    target_frame: str,
    source_frame: str,
    stamp_ns: int | None,
):
    if stamp_ns is None:
        return lookup_transform(buffer, target_frame, source_frame, None)

    offsets_ns = [
        0,
        50_000_000,
        -50_000_000,
        100_000_000,
        -100_000_000,
        250_000_000,
        -250_000_000,
        500_000_000,
        -500_000_000,
        1_000_000_000,
        -1_000_000_000,
        2_000_000_000,
        -2_000_000_000,
    ]
    seen: set[int] = set()
    for offset_ns in offsets_ns:
        candidate_ns = stamp_ns + offset_ns
        if candidate_ns < 0 or candidate_ns in seen:
            continue
        seen.add(candidate_ns)
        transform = lookup_transform(buffer, target_frame, source_frame, candidate_ns)
        if transform is not None:
            return transform
    return None


def transform_distance(transform_a, transform_b) -> float:
    ax = transform_a.transform.translation.x
    ay = transform_a.transform.translation.y
    az = transform_a.transform.translation.z
    bx = transform_b.transform.translation.x
    by = transform_b.transform.translation.y
    bz = transform_b.transform.translation.z
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def get_plug_port_distance(
    buffer, connection: Connection, stamp_ns: int
) -> float | None:
    plug_tf = lookup_transform_with_fallback(
        buffer, "aic_world", connection.plug_tf_name(), stamp_ns
    )
    port_tf = lookup_transform_with_fallback(
        buffer, "aic_world", connection.port_tf_name(), stamp_ns
    )
    if plug_tf is None or port_tf is None:
        return None
    return transform_distance(plug_tf, port_tf)


def find_valid_distance_near_time(
    buffer,
    connection: Connection,
    tf_timestamps_ns: list[int],
    reference_ns: int,
    *,
    forward: bool,
) -> float | None:
    if forward:
        candidates = (stamp for stamp in tf_timestamps_ns if stamp >= reference_ns)
    else:
        candidates = (
            stamp for stamp in reversed(tf_timestamps_ns) if stamp <= reference_ns
        )

    checked = 0
    for candidate_ns in candidates:
        distance = get_plug_port_distance(buffer, connection, candidate_ns)
        if distance is not None:
            return distance
        checked += 1
        if checked >= 200:
            break
    return None


def parse_insertion_namespace(insertion_port_namespace: str) -> list[str]:
    return [token for token in insertion_port_namespace.split("/") if token]


def compute_tier3(
    buffer,
    connection: Connection,
    task_start_ns: int | None,
    task_end_ns: int | None,
    insertion_port_namespace: str,
) -> tuple[float, str]:
    k_insertion_completion_score = 75.0
    k_insertion_penalty = -12.0

    if task_end_ns is None:
        return 0.0, "Task not completed."

    if insertion_port_namespace:
        tokens = parse_insertion_namespace(insertion_port_namespace)
        if len(tokens) >= 2:
            if (
                tokens[0] == connection.target_module_name
                and tokens[1] == connection.port_name
            ):
                return k_insertion_completion_score, "Cable insertion successful."
            return k_insertion_penalty, "Cable insertion failed. Incorrect Port."
        return (
            0.0,
            f"Error parsing insertion port namespace: {insertion_port_namespace}",
        )

    if task_start_ns is None:
        return 0.0, "Distance computation failed, task start time not set"

    init_dist = get_plug_port_distance(buffer, connection, task_start_ns)
    radius_from_port = 0.0 if init_dist is None else init_dist * 0.5

    k_closest_task_score = 25.0
    k_furthest_task_score = 0.0
    k_min_insertion_score = 38.0
    k_max_insertion_score = 50.0
    k_entrance_xy_tol = 0.005

    dist = get_plug_port_distance(buffer, connection, task_end_ns)
    if dist is None:
        return 0.0, "Distance computation failed, tf between cable and port not found"

    port_entrance_tf = lookup_transform_with_fallback(
        buffer, "aic_world", connection.port_entrance_tf_name(), task_end_ns
    )
    port_tf = lookup_transform_with_fallback(
        buffer, "aic_world", connection.port_tf_name(), task_end_ns
    )
    plug_tf = lookup_transform_with_fallback(
        buffer, "aic_world", connection.plug_tf_name(), task_end_ns
    )
    if port_entrance_tf is None or port_tf is None or plug_tf is None:
        return (
            0.0,
            "Distance computation failed, tf between plug, port and port entrance not found",
        )

    port_entrance = port_entrance_tf.transform.translation
    port = port_tf.transform.translation
    plug = plug_tf.transform.translation
    distance_threshold = abs(port_entrance.z - port.z)

    if (
        abs(plug.x - port.x) < k_entrance_xy_tol
        and abs(plug.y - port.y) < k_entrance_xy_tol
        and plug.z < port_entrance.z
        and plug.z - port.z > -0.01
    ):
        port_to_entrance_dist = port_entrance.z - port.z
        plug_to_port_dist = plug.z - port.z
        score = inverse_proportional_score(
            k_max_insertion_score,
            k_min_insertion_score,
            port_to_entrance_dist,
            0.0,
            plug_to_port_dist,
        )
        return (
            score,
            f"Partial insertion detected with distance of {plug_to_port_dist}m.",
        )

    score = inverse_proportional_score(
        k_closest_task_score,
        k_furthest_task_score,
        distance_threshold + radius_from_port,
        distance_threshold,
        dist,
    )
    return score, f"No insertion detected. Final plug port distance: {dist}m."


def compute_insertion_force_score(
    wrenches: list[tuple[float, Vector3]],
) -> CategoryScore:
    k_force_threshold = 20.0
    k_duration_threshold = 1.0
    k_penalty = -12.0

    max_force = 0.0
    time_above_threshold = 0.0
    for index in range(1, len(wrenches)):
        force = wrenches[index][1]
        force_mag = math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z)
        if force_mag > k_force_threshold:
            time_above_threshold += wrenches[index][0] - wrenches[index - 1][0]
        max_force = max(max_force, force_mag)

    if time_above_threshold == 0.0:
        return CategoryScore(0.0, "No excessive force detected")

    message = (
        f"Insertion force above {k_force_threshold:.0f} N, detected for a time of "
        f"{fixed2(time_above_threshold)} seconds. Max detected force: {fixed2(max_force)}N."
    )
    if time_above_threshold > k_duration_threshold:
        return CategoryScore(
            k_penalty,
            message
            + f" This is above the threshold of {fixed2(k_duration_threshold)} seconds. Penalty applied.",
        )
    return CategoryScore(
        0.0,
        message
        + f" This is below the threshold of {fixed2(k_duration_threshold)} seconds. Penalty not applied.",
    )


def compute_contacts_score(contacts) -> CategoryScore:
    if not contacts:
        return CategoryScore(0.0, "No contact detected.")
    first_contact = contacts[0].contacts[0]
    message = (
        "Contacts detected (only first reported) between entity named "
        f"[{first_contact.collision1.name}] and [{first_contact.collision2.name}]. Penalty applied."
    )
    return CategoryScore(-24.0, message)


def compute_task_duration_score(
    tier3_score: float, task_start_ns: int | None, task_end_ns: int | None
) -> CategoryScore:
    k_max_task_time = 60.0
    k_min_task_time = 5.0
    k_fastest_task_score = 12.0
    k_slowest_task_score = 0.0

    if task_end_ns is None:
        return CategoryScore(0.0, "Task not completed.")
    if task_start_ns is None:
        return CategoryScore(0.0, "Time computation failed, task start time not set")
    if tier3_score <= 0.0:
        return CategoryScore(
            0.0,
            "Plug is not within max bounding radius from target port, not assigning time bonus",
        )

    task_duration = bag_time_to_seconds(task_end_ns - task_start_ns)
    score = inverse_proportional_score(
        k_fastest_task_score,
        k_slowest_task_score,
        k_max_task_time,
        k_min_task_time,
        task_duration,
    )
    return CategoryScore(score, f"Task duration: {fixed2(task_duration)} seconds.")


def compute_trajectory_efficiency_score(
    end_effector_poses: list[TransformRecord],
    min_path_length: float,
    tier3_score: float,
    task_end_ns: int | None,
) -> CategoryScore:
    if task_end_ns is None:
        return CategoryScore(0.0, "Task not completed.")
    if tier3_score <= 0.0:
        return CategoryScore(
            0.0,
            "Plug is not within max bounding radius from target port, not assigning efficiency bonus",
        )

    total_path_length = 0.0
    for index in range(1, len(end_effector_poses)):
        prev_pose = end_effector_poses[index - 1]
        pose = end_effector_poses[index]
        dx = pose.x - prev_pose.x
        dy = pose.y - prev_pose.y
        dz = pose.z - prev_pose.z
        total_path_length += math.sqrt(dx * dx + dy * dy + dz * dz)

    score = inverse_proportional_score(
        6.0,
        0.0,
        1.0 + min_path_length,
        min_path_length,
        total_path_length,
    )
    return CategoryScore(
        score,
        "Total end-effector path length: "
        f"{fixed2(total_path_length)} m, initial plug-port distance: {fixed2(min_path_length)} m",
    )


def compute_trajectory_jerk_score(
    end_effector_velocities: list[tuple[float, Vector3]],
    tier3_score: float,
    task_end_ns: int | None,
) -> CategoryScore:
    import numpy as np

    k_window_size = 15
    k_half_window = k_window_size // 2
    k_velocity_threshold = 0.01

    if task_end_ns is None:
        return CategoryScore(0.0, "Task not completed.")
    if tier3_score <= 0.0:
        return CategoryScore(
            0.0,
            "Plug is not within max bounding radius from target port, not assigning jerk bonus",
        )
    if len(end_effector_velocities) < k_window_size:
        return CategoryScore(0.0, "Insufficient velocity samples for jerk computation.")

    total_jerk_time = 0.0
    accum_linear_jerk_magnitude = 0.0

    for index in range(k_half_window, len(end_effector_velocities) - k_half_window):
        velocity = end_effector_velocities[index][1]
        speed = math.sqrt(
            velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z
        )
        if speed <= k_velocity_threshold:
            continue

        center_time = end_effector_velocities[index][0]
        window = end_effector_velocities[
            index - k_half_window : index + k_half_window + 1
        ]
        design = []
        values_x = []
        values_y = []
        values_z = []
        for sample_time, sample_velocity in window:
            dt = sample_time - center_time
            design.append([1.0, dt, dt * dt])
            values_x.append(sample_velocity.x)
            values_y.append(sample_velocity.y)
            values_z.append(sample_velocity.z)

        matrix = np.asarray(design, dtype=np.float64)
        coeffs_x, *_ = np.linalg.lstsq(
            matrix, np.asarray(values_x, dtype=np.float64), rcond=None
        )
        coeffs_y, *_ = np.linalg.lstsq(
            matrix, np.asarray(values_y, dtype=np.float64), rcond=None
        )
        coeffs_z, *_ = np.linalg.lstsq(
            matrix, np.asarray(values_z, dtype=np.float64), rcond=None
        )

        jerk_x = 2.0 * float(coeffs_x[2])
        jerk_y = 2.0 * float(coeffs_y[2])
        jerk_z = 2.0 * float(coeffs_z[2])
        jerk_magnitude = math.sqrt(jerk_x * jerk_x + jerk_y * jerk_y + jerk_z * jerk_z)

        t0 = end_effector_velocities[index - k_half_window][0]
        t1 = end_effector_velocities[index + k_half_window][0]
        dt = (t1 - t0) / k_window_size
        total_jerk_time += dt
        accum_linear_jerk_magnitude += jerk_magnitude * dt

    if abs(total_jerk_time) < 1e-6:
        return CategoryScore(
            0.0, "Error computing jerk. Insufficient end-effector pose samples."
        )

    jerk = accum_linear_jerk_magnitude / total_jerk_time
    score = inverse_proportional_score(6.0, 0.0, 50.0, 0.0, jerk)
    return CategoryScore(
        score,
        f"Average linear jerk magnitude of the end effector: {fixed2(jerk)} m/s^3",
    )


def build_connections(
    trial_cfg: dict[str, Any], task_board_name: str
) -> tuple[list[Connection], int]:
    tasks_cfg = trial_cfg.get("tasks")
    if not isinstance(tasks_cfg, dict) or not tasks_cfg:
        raise ValueError("Trial config does not contain any tasks.")

    connections: list[Connection] = []
    max_time_limit = 0
    for task_cfg in tasks_cfg.values():
        connections.append(
            Connection(
                cable_name=str(task_cfg["cable_name"]),
                task_board_name=task_board_name,
                plug_name=str(task_cfg["plug_name"]),
                target_module_name=str(task_cfg["target_module_name"]),
                port_name=str(task_cfg["port_name"]),
            )
        )
        max_time_limit = max(max_time_limit, int(task_cfg["time_limit"]))

    return connections, max_time_limit


def read_metadata(bag_dir: Path) -> dict[str, Any]:
    return load_yaml(metadata_path_for_bag(bag_dir)).get(
        "rosbag2_bagfile_information", {}
    )


def recover_trial_score(
    bag_dir: Path,
    connections: list[Connection],
    gripper_frame: str,
    buffer_seconds: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import rclpy
    import rosbag2_py
    from geometry_msgs.msg import TransformStamped
    from rclpy.duration import Duration
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from tf2_ros import Buffer

    metadata = read_metadata(bag_dir)
    storage_id = str(metadata.get("storage_identifier", "mcap"))
    starting_time_ns = int(
        metadata.get("starting_time", {}).get("nanoseconds_since_epoch", 0)
    )
    duration_ns = int(metadata.get("duration", {}).get("nanoseconds", 0))

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )

    topics = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_types = {name: get_message(type_name) for name, type_name in topics.items()}

    buffer = Buffer(
        cache_time=Duration(
            seconds=max(buffer_seconds, math.ceil(duration_ns * 1e-9) + 1)
        )
    )
    end_effector_poses: list[TransformRecord] = []
    end_effector_velocities: list[tuple[float, Vector3]] = []
    wrenches: list[tuple[float, Vector3]] = []
    contacts = []
    insertion_port_namespace = ""
    last_tared_ft = None
    first_command_ns: int | None = None
    first_insertion_event_ns: int | None = None
    last_activity_ns: int | None = None
    last_velocity_stamp_s: float | None = None
    tf_timestamps_ns: list[int] = []

    def note_activity(stamp_ns: int) -> None:
        nonlocal last_activity_ns
        if last_activity_ns is None or stamp_ns > last_activity_ns:
            last_activity_ns = stamp_ns

    def latest_end_effector_pose() -> None:
        transform = lookup_transform(buffer, "aic_world", gripper_frame, None)
        if transform is None:
            return
        stamp = (
            transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
        )
        if end_effector_poses and end_effector_poses[-1].stamp_ns == stamp:
            return
        end_effector_poses.append(
            TransformRecord(
                stamp_ns=stamp,
                x=float(transform.transform.translation.x),
                y=float(transform.transform.translation.y),
                z=float(transform.transform.translation.z),
            )
        )

    while reader.has_next():
        topic_name, raw_data, bag_timestamp_ns = reader.read_next()
        if topic_name not in message_types:
            continue

        if topic_name != "/tf_static":
            note_activity(int(bag_timestamp_ns))

        msg = deserialize_message(raw_data, message_types[topic_name])

        if topic_name == "/tf_static":
            for transform in msg.transforms:
                buffer.set_transform_static(transform, "recover_scoring")
        elif topic_name == "/tf":
            for transform in msg.transforms:
                tf_stamp_ns = (
                    transform.header.stamp.sec * 1_000_000_000
                    + transform.header.stamp.nanosec
                )
                if tf_stamp_ns > 0:
                    tf_timestamps_ns.append(tf_stamp_ns)
                if "task_board" in transform.child_frame_id:
                    buffer.set_transform_static(transform, "recover_scoring")
                else:
                    buffer.set_transform(transform, "recover_scoring")
            latest_end_effector_pose()
        elif topic_name == "/scoring/tf":
            for transform in msg.transforms:
                tf_stamp_ns = (
                    transform.header.stamp.sec * 1_000_000_000
                    + transform.header.stamp.nanosec
                )
                if tf_stamp_ns > 0:
                    tf_timestamps_ns.append(tf_stamp_ns)
                buffer.set_transform(transform, "recover_scoring")
            latest_end_effector_pose()
        elif topic_name == "/aic/gazebo/contacts/off_limit":
            if msg.contacts:
                contacts.append(msg)
        elif topic_name == "/fts_broadcaster/wrench":
            if last_tared_ft is not None:
                wrench = Vector3(
                    x=float(msg.wrench.force.x - last_tared_ft.wrench.force.x),
                    y=float(msg.wrench.force.y - last_tared_ft.wrench.force.y),
                    z=float(msg.wrench.force.z - last_tared_ft.wrench.force.z),
                )
                stamp_s = (
                    float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
                )
                wrenches.append((stamp_s, wrench))
        elif topic_name == "/aic_controller/pose_commands":
            if first_command_ns is None:
                first_command_ns = int(bag_timestamp_ns)
        elif topic_name == "/aic_controller/joint_commands":
            if first_command_ns is None:
                first_command_ns = int(bag_timestamp_ns)
        elif topic_name == "/scoring/insertion_event":
            insertion_port_namespace = str(msg.data)
            if first_insertion_event_ns is None:
                first_insertion_event_ns = int(bag_timestamp_ns)
        elif topic_name == "/aic_controller/controller_state":
            last_tared_ft = msg.fts_tare_offset
            stamp_s = (
                float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            )
            if last_velocity_stamp_s == stamp_s:
                continue
            last_velocity_stamp_s = stamp_s
            linear = msg.tcp_velocity.linear
            end_effector_velocities.append(
                (
                    stamp_s,
                    Vector3(x=float(linear.x), y=float(linear.y), z=float(linear.z)),
                )
            )

    identity = TransformStamped()
    identity.header.frame_id = "world"
    identity.child_frame_id = "aic_world"
    identity.transform.rotation.w = 1.0
    buffer.set_transform_static(identity, "recover_scoring")

    task_start_ns = (
        first_command_ns if first_command_ns is not None else starting_time_ns
    )
    task_end_ns = (
        first_insertion_event_ns
        if first_insertion_event_ns is not None
        else last_activity_ns
    )
    tf_timestamps_ns = sorted(set(tf_timestamps_ns))

    primary_connection = connections[0]
    min_path_length = 0.0
    init_dist = (
        get_plug_port_distance(buffer, primary_connection, task_start_ns)
        if task_start_ns
        else None
    )
    if init_dist is None and task_start_ns is not None and tf_timestamps_ns:
        init_dist = find_valid_distance_near_time(
            buffer,
            primary_connection,
            tf_timestamps_ns,
            task_start_ns,
            forward=True,
        )
    if init_dist is not None:
        min_path_length = init_dist

    tier3_score, tier3_message = compute_tier3(
        buffer=buffer,
        connection=primary_connection,
        task_start_ns=task_start_ns,
        task_end_ns=task_end_ns,
        insertion_port_namespace=insertion_port_namespace,
    )

    tier2_categories = {
        "insertion force": compute_insertion_force_score(wrenches),
        "contacts": compute_contacts_score(contacts),
        "duration": compute_task_duration_score(
            tier3_score, task_start_ns, task_end_ns
        ),
        "trajectory smoothness": compute_trajectory_jerk_score(
            end_effector_velocities, tier3_score, task_end_ns
        ),
        "trajectory efficiency": compute_trajectory_efficiency_score(
            end_effector_poses, min_path_length, tier3_score, task_end_ns
        ),
    }
    tier2_score = score_total(tier2_categories)

    tier_1 = tier_to_yaml(1.0, "Model validation succeeded.")
    tier_2 = tier_to_yaml(tier2_score, "Scoring succeeded.", tier2_categories)
    tier_3 = tier_to_yaml(tier3_score, tier3_message)

    debug = {
        "bag_dir": str(bag_dir),
        "task_start_ns": task_start_ns,
        "task_end_ns": task_end_ns,
        "first_command_ns": first_command_ns,
        "first_insertion_event_ns": first_insertion_event_ns,
        "last_activity_ns": last_activity_ns,
        "insertion_port_namespace": insertion_port_namespace,
        "min_path_length": min_path_length,
        "num_end_effector_poses": len(end_effector_poses),
        "num_end_effector_velocities": len(end_effector_velocities),
        "num_wrenches": len(wrenches),
        "num_contacts": len(contacts),
        "used_bag_duration_s": duration_ns * 1e-9,
    }
    if verbose:
        print(
            f"Recovered {bag_dir.name}: start={task_start_ns}, end={task_end_ns}, "
            f"insertion='{insertion_port_namespace}', tier2={tier2_score:.3f}, tier3={tier3_score:.3f}"
        )

    return {"tier_1": tier_1, "tier_2": tier_2, "tier_3": tier_3}, debug


def run_inside_container(args: argparse.Namespace) -> int:
    import rclpy

    results_dir = args.results_dir.resolve()
    scenario_yaml = args.scenario_yaml.resolve()
    output = (args.output or (results_dir / "scoring.yaml")).resolve()

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
    if not scenario_yaml.exists():
        raise FileNotFoundError(f"Scenario YAML does not exist: {scenario_yaml}")
    if output.exists() and args.keep_existing_output:
        raise FileExistsError(f"Output already exists: {output}")

    selected_trials = set(args.trials) if args.trials else None
    scenario = load_yaml(scenario_yaml)
    trials_cfg = scenario.get("trials")
    if not isinstance(trials_cfg, dict) or not trials_cfg:
        raise ValueError(f"No trials found in scenario YAML: {scenario_yaml}")

    bag_dirs, skipped_incomplete = latest_complete_bags(results_dir, selected_trials)
    if not bag_dirs:
        raise FileNotFoundError(
            f"No complete bag_trial_* directories were found under {results_dir}"
        )

    recovered: dict[str, Any] = {}
    debug_data: dict[str, Any] = {}

    rclpy.init()
    try:
        for trial_id in sorted(bag_dirs, key=lambda name: int(name.split("_")[1])):
            if trial_id not in trials_cfg:
                if args.verbose:
                    print(f"Skipping {trial_id}: not present in {scenario_yaml}")
                continue

            connections, max_time_limit = build_connections(
                trials_cfg[trial_id], args.task_board_name
            )
            bag_metadata = read_metadata(bag_dirs[trial_id])
            duration_s = math.ceil(
                int(bag_metadata.get("duration", {}).get("nanoseconds", 0)) * 1e-9
            )
            trial_score, trial_debug = recover_trial_score(
                bag_dir=bag_dirs[trial_id],
                connections=connections,
                gripper_frame=args.gripper_frame,
                buffer_seconds=max(max_time_limit + 5, duration_s + 1),
                verbose=args.verbose,
            )
            recovered[trial_id] = trial_score
            debug_data[trial_id] = trial_debug
    finally:
        rclpy.shutdown()

    total_score = 0.0
    scoring_yaml: dict[str, Any] = {"total": 0.0}
    for trial_id in sorted(recovered, key=lambda name: int(name.split("_")[1])):
        trial_score = recovered[trial_id]
        scoring_yaml[trial_id] = trial_score
        total_score += (
            float(trial_score["tier_1"]["score"])
            + float(trial_score["tier_2"]["score"])
            + float(trial_score["tier_3"]["score"])
        )
    scoring_yaml["total"] = float(total_score)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(scoring_yaml, file, sort_keys=False)

    print(f"Wrote recovered scoring to {output}")
    print(
        f"Recovered trials: {', '.join(sorted(recovered, key=lambda name: int(name.split('_')[1])))}"
    )
    if skipped_incomplete:
        skipped_names = ", ".join(path.name for path in skipped_incomplete)
        print(f"Ignored incomplete bags without metadata.yaml: {skipped_names}")
    if args.verbose:
        print(yaml.safe_dump({"debug": debug_data}, sort_keys=False))
    return 0


def main() -> int:
    args = parse_args()
    if (
        args._run_inside_container
        or os.environ.get("AIC_RECOVER_SCORING_IN_CONTAINER") == "1"
    ):
        return run_inside_container(args)
    return rerun_inside_container(args)


if __name__ == "__main__":
    raise SystemExit(main())
