from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence


TELEOP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = TELEOP_ROOT / "outputs" / "gazebo_teleop"
JOINT_TELEOP = (
    REPO_ROOT
    / "aic_utils"
    / "aic_teleoperation"
    / "aic_teleoperation"
    / "joint_keyboard_teleop.py"
)
CARTESIAN_TELEOP = (
    REPO_ROOT
    / "aic_utils"
    / "aic_teleoperation"
    / "aic_teleoperation"
    / "cartesian_keyboard_teleop.py"
)
EPISODE_STATUS_TOPIC = "/cheatcode/episode_status"
EPISODE_STATUS_TYPE = "std_msgs/msg/String"
HANDOFF_STATUS_TOPIC = "/cheatcode/handoff_status"
HANDOFF_STATUS_TYPE = "std_msgs/msg/String"
HANDOFF_CONTROL_TOPIC = "/cheatcode/handoff_control"
HANDOFF_CONTROL_TYPE = "std_msgs/msg/String"


class GazeboTeleopError(RuntimeError):
    """Raised when the local Gazebo teleop wrapper cannot assemble a valid command."""


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    env_overrides: dict[str, str] | None = None,
    dry_run: bool = False,
) -> int:
    print(f"Running: {_shell_join(command)}")
    if env_overrides:
        for key in sorted(env_overrides):
            print(f"Env: {key}={env_overrides[key]}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


def _pixi_command(*args: str) -> list[str]:
    return ["pixi", "run", "--frozen", *args]


def _teleop_command(script_path: Path, extra_args: Sequence[str]) -> list[str]:
    if not script_path.is_file():
        raise GazeboTeleopError(f"Missing Gazebo teleop script: {script_path}")
    return _pixi_command("python3", str(script_path), *extra_args)


def _cartesian_teleop_args(user_args: Sequence[str]) -> list[str]:
    resolved = list(user_args)
    if any("command_mode:=" in arg for arg in resolved):
        return resolved
    return ["--ros-args", "-p", "command_mode:=position", *resolved]


def _status_publish_command(status: str) -> list[str]:
    payload = json.dumps({"data": status})
    return _pixi_command(
        "ros2",
        "topic",
        "pub",
        "--once",
        EPISODE_STATUS_TOPIC,
        EPISODE_STATUS_TYPE,
        payload,
    )


def _handoff_wait_command() -> list[str]:
    return _pixi_command(
        "ros2",
        "topic",
        "echo",
        "--once",
        HANDOFF_STATUS_TOPIC,
        HANDOFF_STATUS_TYPE,
    )


def _handoff_publish_command(status: str) -> list[str]:
    payload = json.dumps({"data": status})
    return _pixi_command(
        "ros2",
        "topic",
        "pub",
        "--once",
        HANDOFF_CONTROL_TOPIC,
        HANDOFF_CONTROL_TYPE,
        payload,
    )


def _session_name(explicit_name: str | None) -> str:
    return explicit_name or _timestamp()


def _session_root(session_name: str) -> Path:
    return (OUTPUT_ROOT / session_name).resolve()


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _recorder_env(session_name: str) -> tuple[dict[str, str], Path, Path]:
    session_root = _session_root(session_name)
    parquet_root = session_root / "parquet_root"
    rosbag_root = session_root / "bag_root"
    parquet_root.mkdir(parents=True, exist_ok=True)
    rosbag_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["AIC_RECORDER_PARQUET_ROOT"] = str(parquet_root)
    env["AIC_RECORDER_ROSBAG_ROOT"] = str(rosbag_root)
    return env, parquet_root, rosbag_root


def _recorder_command(number_trials: int | None) -> list[str]:
    command = _pixi_command("ros2", "run", "aic_ros_recorder", "aic_ros_recorder_node")
    if number_trials is not None:
        command.extend(["--", "--number-trials", str(number_trials)])
    return command


def _postprocess_command(dataset_root: Path) -> list[str]:
    return _pixi_command(
        "ros2",
        "run",
        "aic_ros_recorder",
        "aic_ros_postprocess_scoring",
        str(dataset_root),
    )


def _add_common_flags(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without launching it.",
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repo-root Gazebo teleop + recording launcher. This wraps the existing "
            "aic_teleoperation and aic_ros_recorder components and keeps outputs under "
            "quantarune_teleop/outputs/gazebo_teleop."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    joint_parser = subparsers.add_parser("joint", help="Run joint-space keyboard teleop.")
    _add_common_flags(joint_parser)
    joint_parser.add_argument(
        "teleop_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying joint teleop script.",
    )

    cartesian_parser = subparsers.add_parser(
        "cartesian", help="Run Cartesian-space keyboard teleop."
    )
    _add_common_flags(cartesian_parser)
    cartesian_parser.add_argument(
        "teleop_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the underlying Cartesian teleop script.",
    )

    record_parser = subparsers.add_parser(
        "record",
        help="Run the AIC ROS recorder with outputs rooted under quantarune_teleop/outputs.",
    )
    _add_common_flags(record_parser)
    record_parser.add_argument(
        "--session-name",
        default=None,
        help="Session directory name under quantarune_teleop/outputs/gazebo_teleop/.",
    )
    record_parser.add_argument(
        "--number-trials",
        type=int,
        default=None,
        help="Auto-stop after this many episodes.",
    )

    start_parser = subparsers.add_parser(
        "episode-start",
        help="Publish a one-shot 'start' control message to the recorder.",
    )
    _add_common_flags(start_parser)

    start_on_handoff_parser = subparsers.add_parser(
        "episode-start-on-handoff",
        help=(
            "Wait for CheatCode handoff_ready on /cheatcode/handoff_status, then "
            "publish the one-shot 'start' control message to the recorder."
        ),
    )
    _add_common_flags(start_on_handoff_parser)
    start_on_handoff_parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="Maximum time to wait for the handoff_ready message before failing.",
    )

    stop_parser = subparsers.add_parser(
        "episode-stop",
        help="Publish a one-shot 'stop' control message to the recorder.",
    )
    _add_common_flags(stop_parser)

    handoff_complete_parser = subparsers.add_parser(
        "handoff-complete",
        help="Publish a one-shot completion message for manual handoff rescue.",
    )
    _add_common_flags(handoff_complete_parser)
    handoff_complete_parser.add_argument(
        "--status",
        default="finish",
        help="Completion status string published to /cheatcode/handoff_control. Default: finish.",
    )

    postprocess_parser = subparsers.add_parser(
        "postprocess-scoring",
        help="Run the recorder scoring postprocess step on an existing dataset root.",
    )
    _add_common_flags(postprocess_parser)
    postprocess_parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to a recorded experiment_<N> dataset root.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)

    if args.command == "joint":
        return _run(
            _teleop_command(JOINT_TELEOP, args.teleop_args),
            dry_run=args.dry_run,
        )

    if args.command == "cartesian":
        return _run(
            _teleop_command(
                CARTESIAN_TELEOP,
                _cartesian_teleop_args(args.teleop_args),
            ),
            dry_run=args.dry_run,
        )

    if args.command == "record":
        session_name = _session_name(args.session_name)
        env, parquet_root, rosbag_root = _recorder_env(session_name)
        session_root = _session_root(session_name)
        _write_manifest(
            session_root / "session_manifest.json",
            {
                "command": "record",
                "created_at": datetime.now().isoformat(),
                "cwd": str(REPO_ROOT),
                "number_trials": args.number_trials,
                "parquet_root": str(parquet_root),
                "rosbag_root": str(rosbag_root),
                "session_name": session_name,
                "teleop_root": str(TELEOP_ROOT),
            },
        )
        return _run(
            _recorder_command(args.number_trials),
            env=env,
            env_overrides={
                "AIC_RECORDER_PARQUET_ROOT": str(parquet_root),
                "AIC_RECORDER_ROSBAG_ROOT": str(rosbag_root),
            },
            dry_run=args.dry_run,
        )

    if args.command == "episode-start":
        return _run(_status_publish_command("start"), dry_run=args.dry_run)

    if args.command == "episode-start-on-handoff":
        if args.dry_run:
            _run(_handoff_wait_command(), dry_run=True)
            return _run(_status_publish_command("start"), dry_run=True)

        print(
            f"Waiting for CheatCode handoff on {HANDOFF_STATUS_TOPIC} "
            f"(timeout {args.timeout_s:.1f}s)..."
        )
        try:
            completed = subprocess.run(
                _handoff_wait_command(),
                cwd=REPO_ROOT,
                check=False,
                timeout=args.timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise GazeboTeleopError(
                f"Timed out waiting for {HANDOFF_STATUS_TOPIC} after {args.timeout_s:.1f}s"
            ) from None

        if completed.returncode != 0:
            return int(completed.returncode)

        print("Received handoff_ready. Starting recorder episode now.")
        return _run(_status_publish_command("start"), dry_run=False)

    if args.command == "episode-stop":
        return _run(_status_publish_command("stop"), dry_run=args.dry_run)

    if args.command == "handoff-complete":
        return _run(_handoff_publish_command(args.status), dry_run=args.dry_run)

    if args.command == "postprocess-scoring":
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        return _run(_postprocess_command(dataset_root), dry_run=args.dry_run)

    parser.error(f"Unsupported command: {args.command}")
    return 2
