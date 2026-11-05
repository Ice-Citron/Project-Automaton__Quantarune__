from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence


DATA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = DATA_ROOT / "outputs" / "gazebo_data"
EPISODE_STATUS_TOPIC = "/cheatcode/episode_status"
EPISODE_STATUS_TYPE = "std_msgs/msg/String"
HANDOFF_CONTROL_TOPIC = "/cheatcode/handoff_control"
HANDOFF_CONTROL_TYPE = "std_msgs/msg/String"
DEFAULT_POLICY = "aic_example_policies.ros.CheatCode"
LOCAL_POLICY_PACKAGE_ROOT = REPO_ROOT / "aic_example_policies"
ZENOH_MODEL_SESSION_SCRIPT = (
    REPO_ROOT / "docker" / "aic_model" / "zenoh_config_model_session.sh"
)


class GazeboDataError(RuntimeError):
    """Raised when the local Gazebo data wrapper cannot assemble a valid command."""


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


def _resolve_pixi_executable() -> str:
    env_override = os.environ.get("PIXI_BIN")
    if env_override:
        return env_override

    resolved = shutil.which("pixi")
    if resolved:
        return resolved

    home = Path(os.environ.get("HOME", str(Path.home())))
    candidates = (
        home / ".pixi" / "bin" / "pixi",
        Path("/home/starforge-sf95/.pixi/bin/pixi"),
        home / ".local" / "bin" / "pixi",
        Path("/usr/local/bin/pixi"),
        Path("/usr/bin/pixi"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise GazeboDataError(
        "Unable to find pixi executable. Set PIXI_BIN or ensure pixi is installed."
    )


def _pixi_command(*args: str) -> list[str]:
    return [_resolve_pixi_executable(), "run", "--frozen", *args]


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


def _policy_command(policy: str, use_sim_time: bool) -> list[str]:
    pixi_bin = shlex.quote(_resolve_pixi_executable())
    sim_time_arg = "-p use_sim_time:=true" if use_sim_time else ""
    shell_command = (
        f"source {shlex.quote(str(ZENOH_MODEL_SESSION_SCRIPT))} && "
        "export ZENOH_ROUTER_CHECK_ATTEMPTS=-1 && "
        f"{pixi_bin} run --as-is env "
        f"PYTHONPATH={shlex.quote(str(LOCAL_POLICY_PACKAGE_ROOT))}:\"${{PYTHONPATH:-}}\" "
        "ros2 run aic_model aic_model --ros-args "
        f"-p policy:={shlex.quote(policy)} {sim_time_arg}"
    ).strip()
    return ["bash", "-lc", shell_command]


def _policy_env_for_mode(
    *,
    mode: str,
    handoff_z_offset_m: float | None = None,
    handoff_settle_s: float | None = None,
    emit_episode_signals: bool | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    env_overrides: dict[str, str] = {"AIC_CHEATCODE_MODE": mode}

    if handoff_z_offset_m is not None:
        env_overrides["AIC_CHEATCODE_HANDOFF_Z_OFFSET_M"] = f"{handoff_z_offset_m}"
    if handoff_settle_s is not None:
        env_overrides["AIC_CHEATCODE_HANDOFF_SETTLE_S"] = f"{handoff_settle_s}"
    if emit_episode_signals is not None:
        env_overrides["AIC_CHEATCODE_EMIT_EPISODE_SIGNALS"] = (
            "true" if emit_episode_signals else "false"
        )

    env.update(env_overrides)
    return env, env_overrides


def _resolve_intervention_mode() -> str:
    mode = os.environ.get("AIC_CHEATCODE_MODE", "handoff").strip().lower()
    if mode not in {"handoff", "auto_rescue"}:
        raise GazeboDataError(
            "Unsupported AIC_CHEATCODE_MODE for intervention-policy: "
            f"{mode!r}. Expected 'handoff' or 'auto_rescue'."
        )
    return mode


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


def _postprocess_command(dataset_root: Path) -> list[str]:
    return _pixi_command(
        "ros2",
        "run",
        "aic_ros_recorder",
        "aic_ros_postprocess_scoring",
        str(dataset_root),
    )


def _peek_command(dataset_root: Path, extra_args: Sequence[str]) -> list[str]:
    peeker = REPO_ROOT / "quantarune_training" / "bash" / "peek_dataset.sh"
    if not peeker.is_file():
        raise GazeboDataError(f"Missing dataset peeker launcher: {peeker}")
    info_json = dataset_root / "meta" / "info.json"
    if not dataset_root.is_dir():
        raise GazeboDataError(f"Dataset root does not exist: {dataset_root}")
    if not info_json.is_file():
        raise GazeboDataError(
            "Dataset root is not a local LeRobot dataset. "
            f"Expected file missing: {info_json}"
        )
    return ["bash", str(peeker), str(dataset_root), *extra_args]


def _add_common_flags(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without launching it.",
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gazebo data collection launcher for cheatcode-driven recording. "
            "This keeps scripted data outputs under quantarune_data/outputs/gazebo_data."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy_parser = subparsers.add_parser(
        "cheatcode",
        help="Run aic_model with the CheatCode policy.",
    )
    _add_common_flags(policy_parser)
    policy_parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help="Policy import path. Defaults to the built-in CheatCode policy.",
    )
    policy_parser.add_argument(
        "--no-sim-time",
        action="store_true",
        help="Disable use_sim_time:=true on the policy node.",
    )

    intervention_parser = subparsers.add_parser(
        "intervention-policy",
        help=(
            "Run CheatCode in pre-insert handoff mode so the scripted phase stops "
            "near the port and the human records only the manual insertion segment."
        ),
    )
    _add_common_flags(intervention_parser)
    intervention_parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help="Policy import path. Defaults to the built-in CheatCode policy.",
    )
    intervention_parser.add_argument(
        "--handoff-z-offset-m",
        type=float,
        default=0.05,
        help="Final scripted stop offset above the port before handing off to teleop.",
    )
    intervention_parser.add_argument(
        "--handoff-settle-s",
        type=float,
        default=1.5,
        help="How long to hold the pre-insert pose before returning control.",
    )
    intervention_parser.add_argument(
        "--no-sim-time",
        action="store_true",
        help="Disable use_sim_time:=true on the policy node.",
    )

    record_parser = subparsers.add_parser(
        "record",
        help="Run the AIC ROS recorder with outputs rooted under quantarune_data/outputs.",
    )
    _add_common_flags(record_parser)
    record_parser.add_argument(
        "--session-name",
        default=None,
        help="Session directory name under quantarune_data/outputs/gazebo_data/.",
    )
    record_parser.add_argument(
        "--number-trials",
        type=int,
        default=None,
        help="Auto-stop after this many recorded episodes.",
    )

    start_parser = subparsers.add_parser(
        "episode-start",
        help="Publish a one-shot 'start' control message to the recorder.",
    )
    _add_common_flags(start_parser)

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

    peek_parser = subparsers.add_parser(
        "peek-dataset",
        help="Replay a recorded dataset through the existing dataset peeker.",
    )
    _add_common_flags(peek_parser)
    peek_parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root to inspect.",
    )
    peek_parser.add_argument(
        "peek_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to peek_dataset.sh after '--'.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "cheatcode":
            return _run(
                _policy_command(args.policy, use_sim_time=not args.no_sim_time),
                dry_run=args.dry_run,
            )

        if args.command == "intervention-policy":
            mode = _resolve_intervention_mode()
            env, env_overrides = _policy_env_for_mode(
                mode=mode,
                handoff_z_offset_m=args.handoff_z_offset_m,
                handoff_settle_s=args.handoff_settle_s,
                emit_episode_signals=(mode == "auto_rescue"),
            )
            return _run(
                _policy_command(args.policy, use_sim_time=not args.no_sim_time),
                env=env,
                env_overrides=env_overrides,
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
                    "data_root": str(DATA_ROOT),
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

        if args.command == "episode-stop":
            return _run(_status_publish_command("stop"), dry_run=args.dry_run)

        if args.command == "handoff-complete":
            return _run(_handoff_publish_command(args.status), dry_run=args.dry_run)

        if args.command == "postprocess-scoring":
            dataset_root = Path(args.dataset_root).expanduser().resolve()
            return _run(_postprocess_command(dataset_root), dry_run=args.dry_run)

        if args.command == "peek-dataset":
            dataset_root = Path(args.dataset_root).expanduser().resolve()
            extra_args = list(args.peek_args)
            if extra_args and extra_args[0] == "--":
                extra_args = extra_args[1:]
            return _run(_peek_command(dataset_root, extra_args), dry_run=args.dry_run)

        parser.error(f"Unsupported command: {args.command}")
        return 2
    except GazeboDataError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
