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


DEFAULT_TASK = "AIC-Task-v0"
TELEOP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_SCRIPT_ROOT = (
    REPO_ROOT / "aic_utils" / "aic_isaac" / "aic_isaaclab" / "scripts"
)
TASK_ROOT = (
    REPO_ROOT
    / "aic_utils"
    / "aic_isaac"
    / "aic_isaaclab"
    / "source"
    / "aic_task"
    / "aic_task"
    / "tasks"
    / "manager_based"
    / "aic_task"
)
INTRINSIC_ASSET_DIR = TASK_ROOT / "Intrinsic_assets"
OUTPUT_ROOT = TELEOP_ROOT / "outputs" / "isaac_teleop"
REQUIRED_ASSET_FILES = (
    INTRINSIC_ASSET_DIR / "scene" / "aic.usd",
    INTRINSIC_ASSET_DIR / "assets" / "Task Board Base" / "task_board_rigid.usd",
    INTRINSIC_ASSET_DIR / "assets" / "SC Plug" / "sc_plug_visual.usd",
    INTRINSIC_ASSET_DIR / "assets" / "SC Port" / "sc_port.usd",
)
SCRIPT_MAP = {
    "check": "list_envs.py",
    "list-envs": "list_envs.py",
    "teleop": "teleop.py",
    "record": "record_demos.py",
    "replay": "replay_demos.py",
}


class IsaacTeleopError(RuntimeError):
    """Raised when the local wrapper cannot assemble a valid Isaac command."""


def _clean_forwarded_args(extra_args: Sequence[str]) -> list[str]:
    forwarded = list(extra_args)
    if forwarded and forwarded[0] == "--":
        return forwarded[1:]
    return forwarded


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _script_path(script_name: str) -> Path:
    path = ISAAC_SCRIPT_ROOT / script_name
    if not path.is_file():
        raise IsaacTeleopError(f"Missing Isaac script: {path}")
    return path


def _find_isaaclab_bin() -> str | None:
    for env_name in ("AIC_ISAACLAB_BIN", "ISAACLAB_BIN"):
        candidate = os.environ.get(env_name)
        if candidate:
            return candidate
    return shutil.which("isaaclab")


def _build_isaaclab_command(
    script_name: str,
    script_args: Sequence[str],
    *,
    allow_missing_bin: bool = False,
) -> list[str]:
    isaaclab_bin = _find_isaaclab_bin()
    if isaaclab_bin is None:
        if allow_missing_bin:
            isaaclab_bin = "isaaclab"
        else:
            raise IsaacTeleopError(
                "Could not find 'isaaclab' on PATH. Set AIC_ISAACLAB_BIN to the "
                "Isaac Lab launcher if needed."
            )
    command = [isaaclab_bin, "-p", str(_script_path(script_name))]
    command.extend(str(arg) for arg in script_args)
    return command


def _run_command(command: Sequence[str], dry_run: bool = False) -> int:
    print(f"Running: {_shell_join(command)}")
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def _latest_dataset_file() -> Path | None:
    if not OUTPUT_ROOT.exists():
        return None
    candidates = sorted(OUTPUT_ROOT.glob("*/dataset.hdf5"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_dataset_file(
    dataset_file: str | None,
    *,
    session_name: str | None = None,
) -> Path:
    if dataset_file:
        return Path(dataset_file).expanduser().resolve()

    resolved_session = session_name or _timestamp()
    return (OUTPUT_ROOT / resolved_session / "dataset.hdf5").resolve()


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _add_common_wrapper_flags(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Isaac Lab command without launching it.",
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repo-local Isaac teleop launcher for the repo-root teleop module. "
            "This wraps the upstream Isaac Lab scripts and keeps outputs in "
            "quantarune_teleop/outputs/isaac_teleop."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check",
        help="Verify local Isaac teleop prerequisites and optionally run list_envs.",
    )
    _add_common_wrapper_flags(check_parser)
    check_parser.add_argument(
        "--run-list-envs",
        action="store_true",
        help="After the local checks pass, launch the upstream list_envs script.",
    )

    list_parser = subparsers.add_parser(
        "list-envs",
        help="Launch the upstream Isaac list_envs entrypoint from this repo.",
    )
    _add_common_wrapper_flags(list_parser)

    teleop_parser = subparsers.add_parser(
        "teleop",
        help="Launch keyboard teleop against AIC-Task-v0 by default.",
    )
    _add_common_wrapper_flags(teleop_parser)
    teleop_parser.add_argument("--task", default=DEFAULT_TASK)
    teleop_parser.add_argument("--teleop-device", default="keyboard")
    teleop_parser.add_argument("--num-envs", type=int, default=1)
    teleop_parser.add_argument(
        "--enable-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Isaac camera sensors for the teleop session.",
    )

    record_parser = subparsers.add_parser(
        "record",
        help="Record teleop demonstrations into an HDF5 dataset under quantarune_teleop/outputs.",
    )
    _add_common_wrapper_flags(record_parser)
    record_parser.add_argument("--task", default=DEFAULT_TASK)
    record_parser.add_argument("--teleop-device", default="keyboard")
    record_parser.add_argument("--num-demos", type=int, default=1)
    record_parser.add_argument("--step-hz", type=int, default=30)
    record_parser.add_argument("--num-success-steps", type=int, default=10)
    record_parser.add_argument(
        "--session-name",
        default=None,
        help="Session directory name under quantarune_teleop/outputs/isaac_teleop/.",
    )
    record_parser.add_argument(
        "--dataset-file",
        default=None,
        help="Optional explicit HDF5 destination. Defaults to outputs/isaac_teleop/<session>/dataset.hdf5.",
    )
    record_parser.add_argument(
        "--enable-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Isaac camera sensors while recording.",
    )

    replay_parser = subparsers.add_parser(
        "replay",
        help="Replay a recorded HDF5 teleop dataset.",
    )
    _add_common_wrapper_flags(replay_parser)
    replay_parser.add_argument("--task", default=None)
    replay_parser.add_argument("--num-envs", type=int, default=1)
    replay_parser.add_argument(
        "--dataset-file",
        default=None,
        help="Recorded HDF5 file to replay.",
    )
    replay_parser.add_argument(
        "--latest",
        action="store_true",
        help="Replay the newest dataset under quantarune_teleop/outputs/isaac_teleop.",
    )
    replay_parser.add_argument(
        "--select-episode",
        action="append",
        dest="select_episodes",
        type=int,
        default=None,
        help="Zero-based episode index to replay. Repeat to select more.",
    )
    replay_parser.add_argument(
        "--validate-states",
        action="store_true",
        help="Pass through to the upstream replay validator.",
    )
    replay_parser.add_argument(
        "--validate-success-rate",
        action="store_true",
        help="Pass through to the upstream replay validator.",
    )
    return parser


def _print_check_summary() -> tuple[bool, list[str]]:
    lines: list[str] = []
    ok = True

    isaaclab_bin = _find_isaaclab_bin()
    if isaaclab_bin is None:
        ok = False
        lines.append("isaaclab: missing from PATH (or set AIC_ISAACLAB_BIN)")
    else:
        lines.append(f"isaaclab: {isaaclab_bin}")

    lines.append(f"repo_root: {REPO_ROOT}")
    lines.append(f"teleop_root: {TELEOP_ROOT}")
    lines.append(f"outputs_root: {OUTPUT_ROOT}")

    missing_scripts: list[Path] = []
    for script_name in ("list_envs.py", "teleop.py", "record_demos.py", "replay_demos.py"):
        script_path = ISAAC_SCRIPT_ROOT / script_name
        if script_path.is_file():
            lines.append(f"script_ok: {script_path}")
        else:
            ok = False
            missing_scripts.append(script_path)

    if missing_scripts:
        lines.append("missing_scripts:")
        lines.extend(f"  - {path}" for path in missing_scripts)

    missing_assets = [path for path in REQUIRED_ASSET_FILES if not path.exists()]
    if missing_assets:
        ok = False
        lines.append(f"asset_dir_missing_or_incomplete: {INTRINSIC_ASSET_DIR}")
        lines.append("missing_asset_files:")
        lines.extend(f"  - {path}" for path in missing_assets)
    else:
        lines.append(f"asset_dir_ok: {INTRINSIC_ASSET_DIR}")

    latest_dataset = _latest_dataset_file()
    if latest_dataset is not None:
        lines.append(f"latest_dataset: {latest_dataset}")
    else:
        lines.append("latest_dataset: none yet")

    return ok, lines


def _handle_check(args: argparse.Namespace) -> int:
    ok, lines = _print_check_summary()
    for line in lines:
        print(line)

    if not ok:
        return 1

    if not args.run_list_envs:
        return 0

    command = _build_isaaclab_command(
        SCRIPT_MAP["check"], [], allow_missing_bin=args.dry_run
    )
    return _run_command(command, dry_run=args.dry_run)


def _handle_list_envs(args: argparse.Namespace, extra_args: Sequence[str]) -> int:
    command = _build_isaaclab_command(
        SCRIPT_MAP["list-envs"], extra_args, allow_missing_bin=args.dry_run
    )
    return _run_command(command, dry_run=args.dry_run)


def _handle_teleop(args: argparse.Namespace, extra_args: Sequence[str]) -> int:
    script_args: list[str] = [
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--teleop_device",
        args.teleop_device,
    ]
    if args.enable_cameras:
        script_args.append("--enable_cameras")
    script_args.extend(extra_args)

    command = _build_isaaclab_command(
        SCRIPT_MAP["teleop"], script_args, allow_missing_bin=args.dry_run
    )
    return _run_command(command, dry_run=args.dry_run)


def _handle_record(args: argparse.Namespace, extra_args: Sequence[str]) -> int:
    dataset_file = _resolve_dataset_file(
        args.dataset_file,
        session_name=args.session_name,
    )
    session_dir = dataset_file.parent

    script_args: list[str] = [
        "--task",
        args.task,
        "--teleop_device",
        args.teleop_device,
        "--dataset_file",
        str(dataset_file),
        "--step_hz",
        str(args.step_hz),
        "--num_demos",
        str(args.num_demos),
        "--num_success_steps",
        str(args.num_success_steps),
    ]
    if args.enable_cameras:
        script_args.append("--enable_cameras")
    script_args.extend(extra_args)

    command = _build_isaaclab_command(
        SCRIPT_MAP["record"], script_args, allow_missing_bin=args.dry_run
    )
    manifest = {
        "command": command,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(dataset_file),
        "intrinsic_asset_dir": str(INTRINSIC_ASSET_DIR),
        "repo_root": str(REPO_ROOT),
        "task": args.task,
        "teleop_device": args.teleop_device,
        "teleop_root": str(TELEOP_ROOT),
    }
    if not args.dry_run:
        session_dir.mkdir(parents=True, exist_ok=True)
        _write_manifest(session_dir / "session_manifest.json", manifest)
    return _run_command(command, dry_run=args.dry_run)


def _handle_replay(args: argparse.Namespace, extra_args: Sequence[str]) -> int:
    dataset_file: Path | None = None
    if args.latest:
        dataset_file = _latest_dataset_file()
        if dataset_file is None:
            raise IsaacTeleopError(
                "No recorded datasets found under quantarune_teleop/outputs/isaac_teleop."
            )
    elif args.dataset_file:
        dataset_file = Path(args.dataset_file).expanduser().resolve()

    if dataset_file is None:
        raise IsaacTeleopError(
            "Replay needs --dataset-file or --latest."
        )

    script_args: list[str] = [
        "--dataset_file",
        str(dataset_file),
        "--num_envs",
        str(args.num_envs),
    ]
    if args.task:
        script_args.extend(["--task", args.task])
    if args.select_episodes:
        script_args.append("--select_episodes")
        script_args.extend(str(index) for index in args.select_episodes)
    if args.validate_states:
        script_args.append("--validate_states")
    if args.validate_success_rate:
        script_args.append("--validate_success_rate")
    script_args.extend(extra_args)

    command = _build_isaaclab_command(
        SCRIPT_MAP["replay"], script_args, allow_missing_bin=args.dry_run
    )
    return _run_command(command, dry_run=args.dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_parser()
    args, extra_args = parser.parse_known_args(argv)
    forwarded_args = _clean_forwarded_args(extra_args)

    try:
        if args.command == "check":
            return _handle_check(args)
        if args.command == "list-envs":
            return _handle_list_envs(args, forwarded_args)
        if args.command == "teleop":
            return _handle_teleop(args, forwarded_args)
        if args.command == "record":
            return _handle_record(args, forwarded_args)
        if args.command == "replay":
            return _handle_replay(args, forwarded_args)
    except IsaacTeleopError as exc:
        print(f"error: {exc}")
        return 2

    parser.print_help()
    return 1
