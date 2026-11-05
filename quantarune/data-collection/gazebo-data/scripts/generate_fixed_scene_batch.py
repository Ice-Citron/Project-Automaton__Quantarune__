#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = DATA_ROOT / "outputs" / "fixed_scene_batches"
ENGINE_TEMPLATE_PATH = REPO_ROOT / "aic_engine" / "config" / "sample_config.yaml"
BATCH_GAZEBO_DATA_DIRNAME = "gazebo_data"


@dataclass(frozen=True)
class ModeSpec:
    mode: str
    task_args: tuple[str, ...]


MODE_SPECS = {
    "sfp": ModeSpec(
        mode="sfp",
        task_args=(
            "--task-id",
            "fixed_scene_sfp",
            "--cable-type",
            "sfp_sc",
            "--cable-name",
            "cable_0",
            "--plug-type",
            "sfp",
            "--plug-name",
            "sfp_tip",
            "--port-type",
            "sfp",
            "--port-name",
            "sfp_port_0",
            "--target-module-name",
            "nic_card_mount_0",
            "--time-limit",
            "180",
        ),
    ),
    "sc": ModeSpec(
        mode="sc",
        task_args=(
            "--task-id",
            "fixed_scene_sc",
            "--cable-type",
            "sfp_sc",
            "--cable-name",
            "cable_1",
            "--plug-type",
            "sc",
            "--plug-name",
            "sc_tip",
            "--port-type",
            "sc",
            "--port-name",
            "sc_port_base",
            "--target-module-name",
            "sc_port_0",
            "--time-limit",
            "180",
        ),
    ),
}


@dataclass(frozen=True)
class SceneVariant:
    env_overrides: dict[str, str]
    task_args: tuple[str, ...]
    target_name: str
    rollout_profile: str = "clean"
    perturbation_profile: str = "none"
    policy_env_overrides: dict[str, str] | None = None
    planned_demo_bucket: str = "clean_nominal_candidate"
    expected_primary_stage: str = "approach"
    recommended_collection_mode: str = "scripted"
    collection_notes: str = ""


@dataclass(frozen=True)
class PerturbationSpec:
    profile: str
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    decay_end_z_offset_m: float

    def env_overrides(self) -> dict[str, str]:
        return {
            "AIC_CHEATCODE_PERTURB_X_M": _fmt(self.x_m, digits=6),
            "AIC_CHEATCODE_PERTURB_Y_M": _fmt(self.y_m, digits=6),
            "AIC_CHEATCODE_PERTURB_Z_M": _fmt(self.z_m, digits=6),
            "AIC_CHEATCODE_PERTURB_ROLL_RAD": _fmt(self.roll_rad, digits=6),
            "AIC_CHEATCODE_PERTURB_PITCH_RAD": _fmt(self.pitch_rad, digits=6),
            "AIC_CHEATCODE_PERTURB_YAW_RAD": _fmt(self.yaw_rad, digits=6),
            "AIC_CHEATCODE_PERTURB_DECAY_END_Z_OFFSET_M": _fmt(
                self.decay_end_z_offset_m,
                digits=6,
            ),
        }


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a batch of fixed-scene CheatCode cases, optionally recording each "
            "case and auditing the resulting datasets into failure buckets."
        )
    )
    parser.add_argument("--mode", choices=sorted(MODE_SPECS), required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "RNG seed for board/layout randomization. If omitted, a timestamp-derived "
            "seed is used so repeated runs do not silently duplicate the same cases."
        ),
    )
    parser.add_argument(
        "--scene-profile",
        choices=("minimal", "generalization"),
        default="generalization",
        help="Board clutter profile for each generated case. Default: generalization.",
    )
    parser.add_argument(
        "--nic-count-profile",
        choices=("minimal", "qualification_mix", "all"),
        default="qualification_mix",
        help=(
            "How many NIC cards to spawn per case. "
            "'qualification_mix' varies clutter count per trial, "
            "'all' forces all 5 cards, "
            "'minimal' keeps only the minimum cards needed."
        ),
    )
    parser.add_argument(
        "--sfp-target-pool",
        choices=("known_good", "all"),
        default="all",
        help=(
            "Which NIC target set to sample for SFP cases. "
            "'known_good' restricts targets to NIC 0/1, "
            "'all' allows NIC 0..4."
        ),
    )
    parser.add_argument(
        "--rollout-profile",
        choices=("clean", "perturb_sc", "perturb_sfp"),
        default="clean",
        help=(
            "Teacher rollout profile. 'clean' preserves the existing scripted lane. "
            "'perturb_sc' injects controlled pre-insert pose errors for SC. "
            "'perturb_sfp' injects controlled pre-insert pose errors for SFP."
        ),
    )
    parser.add_argument(
        "--perturbation-profile",
        choices=("micro", "hover", "mild", "medium", "coarse", "mixed"),
        default="mild",
        help=(
            "Perturbation strength profile when --rollout-profile perturb_sc is used. "
            "Default: mild."
        ),
    )
    parser.add_argument(
        "--session-prefix",
        help="Prefix for batch and case session names. Defaults to fixed_scene_<mode>_<timestamp>.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not start the recorder for each case.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Only generate per-case configs plus cases.csv/summary.json metadata. "
            "Skip Gazebo, policy, recorder, and audit execution."
        ),
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Do not run dataset bucketing on recorded outputs.",
    )
    parser.add_argument(
        "--sc-demo-bucket-profile",
        choices=("default", "recovery_contact_mix", "primitive_tiered"),
        default="default",
        help=(
            "SC-only collection intent labeling. "
            "'recovery_contact_mix' tags cases as alignment/contact/retry candidates "
            "for later replay into new SC demos. "
            "'primitive_tiered' follows the SC port-centering/hover/align/terminal "
            "bucket structure. Default: default."
        ),
    )
    parser.add_argument(
        "--sfp-demo-bucket-profile",
        choices=("default", "align_guarded_reseat", "recovery_contact_mix", "primitive_tiered"),
        default="default",
        help=(
            "SFP-only collection intent labeling. "
            "'align_guarded_reseat' cycles align, guarded insert, and partial-reseat "
            "candidate buckets for the first controlled SFP pass. "
            "'recovery_contact_mix' also includes side-contact/backout candidates. "
            "'primitive_tiered' prioritizes SFP port-centering, hover, and align "
            "with guarded/reseat as Tier 2. "
            "Default: default."
        ),
    )
    parser.add_argument(
        "--engine-wait",
        type=float,
        default=12.0,
        help="Seconds to wait after Gazebo bringup for each case.",
    )
    parser.add_argument(
        "--policy-wait",
        type=float,
        default=4.0,
        help="Seconds to wait after policy startup for each case.",
    )
    parser.add_argument(
        "--keep-case-stacks",
        action="store_true",
        help="Forward --keep-running to the per-case launcher. Mainly for debugging; not recommended for batch runs.",
    )
    parser.add_argument(
        "--policy-import",
        help=(
            "Policy import path to run instead of the default CheatCode. "
            "Example: aic_example_policies.ros.CheatCodeCableAwareV2"
        ),
    )
    parser.add_argument(
        "--export-preview",
        action="store_true",
        help="Export MP4 previews from each recorded case into the batch directory.",
    )
    parser.add_argument(
        "--preview-max-frames",
        type=int,
        default=600,
        help="Maximum frames per exported preview video. Default: 600.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.75,
        help="Scale factor for exported preview montages. Default: 0.75.",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.rollout_profile == "perturb_sc" and args.mode != "sc":
        parser.error("--rollout-profile perturb_sc currently supports only --mode sc")
    if args.rollout_profile == "perturb_sfp" and args.mode != "sfp":
        parser.error("--rollout-profile perturb_sfp currently supports only --mode sfp")
    if args.sc_demo_bucket_profile != "default" and args.mode != "sc":
        parser.error("--sc-demo-bucket-profile currently supports only --mode sc")
    if args.sfp_demo_bucket_profile != "default" and args.mode != "sfp":
        parser.error("--sfp-demo-bucket-profile currently supports only --mode sfp")
    return args


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_yaml(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return yaml.safe_load(path.read_text()) or {}


def dataset_has_episode_parquet(dataset_root: Path) -> bool:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        return False
    return any(data_dir.rglob("*.parquet"))


def dataset_has_readable_parquet(dataset_root: Path) -> bool:
    data_dir = dataset_root / "data"
    parquet_paths = sorted(data_dir.rglob("*.parquet"))
    if not parquet_paths:
        return False

    python_bin = REPO_ROOT / "quantarune_training" / ".pixi" / "envs" / "default" / "bin" / "python"
    if not python_bin.is_file():
        return True

    command = [
        str(python_bin),
        "-c",
        (
            "import sys; "
            "import pyarrow.parquet as pq; "
            "pq.read_schema(sys.argv[1]); "
            "print('ok')"
        ),
        str(parquet_paths[0]),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def wait_for_dataset_materialization(
    dataset_root: Path,
    *,
    timeout_s: float = 20.0,
    poll_s: float = 0.5,
) -> tuple[bool, bool]:
    """Wait briefly for recorder finalization to materialize parquet output.

    The recorder can still be flushing videos/parquet after the outer launcher
    returns. Poll here so we do not misclassify valid runs as empty_dataset.
    """

    deadline = time.monotonic() + timeout_s
    saw_any_parquet = False

    while time.monotonic() < deadline:
        has_data = dataset_has_episode_parquet(dataset_root)
        if has_data:
            saw_any_parquet = True
            if dataset_has_readable_parquet(dataset_root):
                return True, True
        time.sleep(poll_s)

    if saw_any_parquet:
        return True, dataset_has_readable_parquet(dataset_root)
    return False, False


def relocate_recorded_session(
    *,
    case_name: str,
    batch_root: Path,
) -> tuple[Path, Path]:
    source_session_root = DATA_ROOT / "outputs" / "gazebo_data" / case_name
    target_sessions_root = ensure_dir(batch_root / BATCH_GAZEBO_DATA_DIRNAME)
    target_session_root = target_sessions_root / case_name

    if not source_session_root.exists():
        return source_session_root, source_session_root / "parquet_root" / "experiment_1"

    if target_session_root.exists():
        shutil.rmtree(target_session_root)

    shutil.move(str(source_session_root), str(target_session_root))
    return target_session_root, target_session_root / "parquet_root" / "experiment_1"


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return int(datetime.now().strftime("%Y%m%d%H%M%S"))


def _sample_nic_present_indices(
    *,
    mode: str,
    scene_profile: str,
    nic_count_profile: str,
    rng: random.Random,
    target_index: int | None,
) -> set[int]:
    if scene_profile == "minimal" or nic_count_profile == "minimal":
        if mode == "sfp" and target_index is not None:
            return {target_index}
        return set()

    if nic_count_profile == "all":
        return set(range(5))

    if mode == "sfp":
        count_options = (1, 2, 3, 4, 5)
        count_weights = (0.10, 0.15, 0.20, 0.15, 0.40)
    else:
        count_options = (0, 1, 2, 3, 4, 5)
        count_weights = (0.10, 0.10, 0.15, 0.20, 0.15, 0.30)

    nic_count = rng.choices(count_options, weights=count_weights, k=1)[0]
    if mode == "sfp":
        nic_count = max(1, nic_count)

    indices = set()
    if target_index is not None:
        indices.add(target_index)

    available = [idx for idx in range(5) if idx not in indices]
    remaining = max(0, nic_count - len(indices))
    if remaining > 0:
        indices.update(rng.sample(available, k=remaining))
    return indices


def _deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)


def _sample_signed_abs(
    rng: random.Random,
    min_abs: float,
    max_abs: float,
) -> float:
    magnitude = rng.uniform(min_abs, max_abs)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return sign * magnitude


def _sample_xy_offset(
    rng: random.Random,
    min_radius_m: float,
    max_radius_m: float,
) -> tuple[float, float]:
    radius = rng.uniform(min_radius_m, max_radius_m)
    theta = rng.uniform(-math.pi, math.pi)
    return radius * math.cos(theta), radius * math.sin(theta)


def _sample_sc_perturbation(
    profile: str,
    rng: random.Random,
) -> PerturbationSpec:
    resolved_profile = profile
    if profile == "mixed":
        resolved_profile = rng.choice(("mild", "medium"))

    if resolved_profile == "micro":
        xy_min, xy_max = 0.0005, 0.0015
        z_min, z_max = 0.0015, 0.0040
        yaw_min_deg, yaw_max_deg = 0.75, 2.5
        tilt_min_deg, tilt_max_deg = 0.0, 0.75
        decay_end = 0.010
    elif resolved_profile == "hover":
        xy_min, xy_max = 0.0040, 0.0150
        z_min, z_max = 0.0020, 0.0080
        yaw_min_deg, yaw_max_deg = 0.5, 3.0
        tilt_min_deg, tilt_max_deg = 0.0, 1.0
        decay_end = 0.050
    elif resolved_profile == "mild":
        xy_min, xy_max = 0.0010, 0.0030
        z_min, z_max = 0.0030, 0.0080
        yaw_min_deg, yaw_max_deg = 2.0, 5.0
        tilt_min_deg, tilt_max_deg = 0.5, 1.5
        decay_end = 0.015
    elif resolved_profile == "medium":
        xy_min, xy_max = 0.0025, 0.0050
        z_min, z_max = 0.0080, 0.0150
        yaw_min_deg, yaw_max_deg = 4.0, 8.0
        tilt_min_deg, tilt_max_deg = 1.5, 3.0
        decay_end = 0.020
    elif resolved_profile == "coarse":
        xy_min, xy_max = 0.0250, 0.0700
        z_min, z_max = 0.0040, 0.0150
        yaw_min_deg, yaw_max_deg = 1.0, 6.0
        tilt_min_deg, tilt_max_deg = 0.0, 1.5
        decay_end = 0.060
    else:
        raise ValueError(f"Unsupported perturbation profile: {profile}")

    x_m, y_m = _sample_xy_offset(rng, xy_min, xy_max)
    z_m = rng.uniform(z_min, z_max)
    yaw_rad = _deg_to_rad(_sample_signed_abs(rng, yaw_min_deg, yaw_max_deg))

    # Keep tilt optional so the first perturbation phase stays recoverable.
    if rng.random() < 0.5:
        roll_rad = 0.0
    else:
        roll_rad = _deg_to_rad(_sample_signed_abs(rng, tilt_min_deg, tilt_max_deg))
    if rng.random() < 0.5:
        pitch_rad = 0.0
    else:
        pitch_rad = _deg_to_rad(_sample_signed_abs(rng, tilt_min_deg, tilt_max_deg))

    return PerturbationSpec(
        profile=resolved_profile,
        x_m=x_m,
        y_m=y_m,
        z_m=z_m,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        decay_end_z_offset_m=decay_end,
    )


def _planned_sc_demo_bucket_spec(
    *,
    case_index: int,
    rng: random.Random,
    profile: str = "recovery_contact_mix",
) -> dict[str, str]:
    if profile == "primitive_tiered":
        bucket_cycle = (
            "sc_port_centering_success",
            "sc_hover_acquire_success",
            "sc_align_perturb_success",
            "sc_port_centering_success",
            "sc_hover_acquire_success",
            "sc_align_perturb_success",
            "sc_guarded_insert_contact_success",
            "sc_partial_insert_reseat_success",
            "sc_side_contact_backout_success",
        )
    elif profile == "recovery_contact_mix":
        bucket_cycle = (
            "align_perturb_candidate",
            "side_contact_backout_candidate",
            "partial_insert_reseat_candidate",
            "guarded_insert_contact_candidate",
        )
    else:
        raise ValueError(f"Unsupported SC demo bucket profile: {profile}")
    bucket = bucket_cycle[(case_index - 1) % len(bucket_cycle)]

    if bucket == "sc_port_centering_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "port_centering",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "coarse",
            "collection_notes": (
                "SC Tier 1 port-centering primitive. Start with realistic several-cm "
                "lateral miss at safe standoff, reduce into hover-worthy residual, "
                "and terminate without contact."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "port_centering",
                "AIC_CHEATCODE_PORT_CENTERING_STOP_Z_OFFSET_M": "0.060",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.5",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.0008",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.05",
            },
        }
    if bucket == "sc_hover_acquire_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "hover_acquire",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "hover",
            "collection_notes": (
                "SC Tier 1 hover-acquire primitive. Start near-centered but imperfect, "
                "move to stable low-force hover, dwell, and stop before insertion."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "hover_acquire",
                "AIC_CHEATCODE_HOVER_Z_OFFSET_M": "0.050",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.5",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.0005",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.05",
            },
        }
    if bucket == "sc_align_perturb_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "align",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SC Tier 1 align primitive. Start close with small lateral/angular "
                "perturbation, correct into insert-ready envelope, then stop."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "align_perturb",
                "AIC_CHEATCODE_ALIGN_STAGE_END_Z_OFFSET_M": "0.035",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.0",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.00035",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.06",
            },
        }
    if bucket == "sc_guarded_insert_contact_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "guarded_insert",
            "recommended_collection_mode": "scripted_or_handoff",
            "perturbation_profile": "micro",
            "collection_notes": (
                "SC Tier 2 guarded insert primitive. Start from valid align/hover and "
                "progress through controlled final contact without force violation."
            ),
        }
    if bucket == "sc_partial_insert_reseat_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SC Tier 2 partial-insert reseat primitive. Demonstrate shallow "
                "engagement, small backout/reseat, and safe finish."
            ),
        }
    if bucket == "sc_side_contact_backout_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "mild" if rng.random() < 0.5 else "medium",
            "collection_notes": (
                "SC Tier 2 side-contact/backout primitive. Demonstrate brief side "
                "contact, stop, backout, correction, and retry."
            ),
        }

    if bucket == "align_perturb_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "align",
            "recommended_collection_mode": "scripted_or_handoff",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "Near-port alignment correction candidate. "
                "Target lateral/yaw correction before sustained contact."
            ),
        }
    if bucket == "side_contact_backout_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "mild" if rng.random() < 0.5 else "medium",
            "collection_notes": (
                "Side-contact and backout candidate. "
                "Target brief side contact, backout, and corrected reinsertion."
            ),
        }
    if bucket == "partial_insert_reseat_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "Shallow partial-insert reseat candidate. "
                "Target shallow engagement followed by reseat and finish."
            ),
        }
    return {
        "planned_demo_bucket": "guarded_insert_contact_candidate",
        "expected_primary_stage": "guarded_insert",
        "recommended_collection_mode": "scripted_or_handoff",
        "perturbation_profile": "micro",
        "collection_notes": (
            "Guarded final-phase contact candidate. "
            "Target real last-centimeter contact rather than clean air-only descent."
        ),
    }


def _planned_sfp_demo_bucket_spec(
    *,
    case_index: int,
    rng: random.Random,
    profile: str,
) -> dict[str, str]:
    if profile == "primitive_tiered":
        bucket_cycle = (
            "sfp_port_centering_success",
            "sfp_hover_acquire_success",
            "sfp_align_perturb_success",
            "sfp_port_centering_success",
            "sfp_hover_acquire_success",
            "sfp_align_perturb_success",
            "sfp_guarded_insert_contact_success",
            "sfp_partial_insert_reseat_success",
        )
    elif profile == "align_guarded_reseat":
        bucket_cycle = (
            "sfp_align_perturb_candidate",
            "sfp_guarded_insert_contact_candidate",
            "sfp_partial_insert_reseat_candidate",
        )
    elif profile == "recovery_contact_mix":
        bucket_cycle = (
            "sfp_align_perturb_candidate",
            "sfp_guarded_insert_contact_candidate",
            "sfp_partial_insert_reseat_candidate",
            "sfp_side_contact_backout_candidate",
        )
    else:
        raise ValueError(f"Unsupported SFP demo bucket profile: {profile}")

    bucket = bucket_cycle[(case_index - 1) % len(bucket_cycle)]

    if bucket == "sfp_port_centering_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "port_centering",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "coarse",
            "collection_notes": (
                "SFP Tier 1 port-centering primitive. Start with realistic several-cm "
                "lateral miss at safe standoff, reduce into hover-worthy residual, "
                "and terminate without contact."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "port_centering",
                "AIC_CHEATCODE_PORT_CENTERING_STOP_Z_OFFSET_M": "0.060",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.5",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.0008",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.05",
            },
        }
    if bucket == "sfp_hover_acquire_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "hover_acquire",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "hover",
            "collection_notes": (
                "SFP Tier 1 hover-acquire primitive. Start near-centered but imperfect, "
                "move to stable hover, dwell, and stop before insertion."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "hover_acquire",
                "AIC_CHEATCODE_HOVER_Z_OFFSET_M": "0.050",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.5",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.0005",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.05",
            },
        }
    if bucket == "sfp_align_perturb_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "align",
            "recommended_collection_mode": "scripted_primitive",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SFP Tier 1 align primitive. Start close with small lateral/angular "
                "perturbation, correct into insert-ready envelope, then stop."
            ),
            "policy_env_overrides": {
                "AIC_CHEATCODE_PRIMITIVE_STOP_STAGE": "align_perturb",
                "AIC_CHEATCODE_ALIGN_STAGE_END_Z_OFFSET_M": "0.035",
                "AIC_CHEATCODE_PRIMITIVE_DWELL_S": "1.0",
                "AIC_CHEATCODE_DESCENT_STEP_M": "0.00035",
                "AIC_CHEATCODE_DESCENT_SLEEP_S": "0.06",
            },
        }
    if bucket == "sfp_guarded_insert_contact_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "guarded_insert",
            "recommended_collection_mode": "scripted_or_handoff",
            "perturbation_profile": "micro",
            "collection_notes": (
                "SFP Tier 2 guarded insert primitive. Use only force-safe final-contact "
                "successes for training."
            ),
        }
    if bucket == "sfp_partial_insert_reseat_success":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SFP Tier 2 partial-insert reseat primitive. Demonstrate shallow "
                "engagement followed by a small backout/reseat."
            ),
        }

    if bucket == "sfp_align_perturb_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "align",
            "recommended_collection_mode": "scripted_or_handoff",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SFP near-port alignment correction candidate. "
                "Target stable hover and lateral/yaw correction before final contact."
            ),
        }
    if bucket == "sfp_guarded_insert_contact_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "guarded_insert",
            "recommended_collection_mode": "scripted_or_handoff",
            "perturbation_profile": "micro",
            "collection_notes": (
                "SFP guarded final-phase contact candidate. "
                "Target controlled last-centimeter insertion under realistic light contact."
            ),
        }
    if bucket == "sfp_partial_insert_reseat_candidate":
        return {
            "planned_demo_bucket": bucket,
            "expected_primary_stage": "retry_or_finish",
            "recommended_collection_mode": "handoff_or_manual_rescue",
            "perturbation_profile": "micro" if rng.random() < 0.5 else "mild",
            "collection_notes": (
                "SFP shallow partial-insert reseat candidate. "
                "Target shallow engagement followed by a small backout/reseat and finish."
            ),
        }
    return {
        "planned_demo_bucket": "sfp_side_contact_backout_candidate",
        "expected_primary_stage": "retry_or_finish",
        "recommended_collection_mode": "handoff_or_manual_rescue",
        "perturbation_profile": "mild" if rng.random() < 0.5 else "medium",
        "collection_notes": (
            "SFP side-contact/backout candidate. "
            "Only prioritize this bucket if audit shows side/rim contact is a real SFP failure mode."
        ),
    }


def _env_bool(env: dict[str, str], key: str, default: bool = False) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(env: dict[str, str], key: str, default: float = 0.0) -> float:
    raw = env.get(key)
    if raw is None:
        return default
    return float(raw)


def load_engine_template() -> dict[str, object]:
    return yaml.safe_load(ENGINE_TEMPLATE_PATH.read_text())


def build_engine_trial(mode: str, scene_variant: SceneVariant) -> dict[str, object]:
    env = scene_variant.env_overrides

    task_board = {
        "pose": {
            "x": _env_float(env, "TASK_BOARD_X"),
            "y": _env_float(env, "TASK_BOARD_Y"),
            "z": _env_float(env, "TASK_BOARD_Z", 1.14),
            "roll": _env_float(env, "TASK_BOARD_ROLL"),
            "pitch": _env_float(env, "TASK_BOARD_PITCH"),
            "yaw": _env_float(env, "TASK_BOARD_YAW"),
        }
    }

    for idx in range(5):
        key = f"nic_rail_{idx}"
        task_board[key] = {
            "entity_present": _env_bool(env, f"NIC_CARD_MOUNT_{idx}_PRESENT"),
        }
        if task_board[key]["entity_present"]:
            task_board[key]["entity_name"] = f"nic_card_{idx}"
            task_board[key]["entity_pose"] = {
                "translation": _env_float(env, f"NIC_CARD_MOUNT_{idx}_TRANSLATION"),
                "roll": _env_float(env, f"NIC_CARD_MOUNT_{idx}_ROLL"),
                "pitch": _env_float(env, f"NIC_CARD_MOUNT_{idx}_PITCH"),
                "yaw": _env_float(env, f"NIC_CARD_MOUNT_{idx}_YAW"),
            }

    for idx in range(2):
        key = f"sc_rail_{idx}"
        task_board[key] = {
            "entity_present": _env_bool(env, f"SC_PORT_{idx}_PRESENT"),
        }
        if task_board[key]["entity_present"]:
            task_board[key]["entity_name"] = f"sc_mount_{idx}"
            task_board[key]["entity_pose"] = {
                "translation": _env_float(env, f"SC_PORT_{idx}_TRANSLATION"),
                "roll": _env_float(env, f"SC_PORT_{idx}_ROLL"),
                "pitch": _env_float(env, f"SC_PORT_{idx}_PITCH"),
                "yaw": _env_float(env, f"SC_PORT_{idx}_YAW"),
            }

    for key in (
        "lc_mount_rail_0",
        "lc_mount_rail_1",
        "sfp_mount_rail_0",
        "sfp_mount_rail_1",
        "sc_mount_rail_0",
        "sc_mount_rail_1",
    ):
        task_board[key] = {"entity_present": False}

    if mode == "sc":
        cable_name = "cable_1"
        cable_type = "sfp_sc_cable_reversed"
        plug_type = "sc"
        plug_name = "sc_tip"
        port_type = "sc"
        port_name = "sc_port_base"
        gripper_z = 0.04045
    else:
        cable_name = "cable_0"
        cable_type = "sfp_sc_cable"
        plug_type = "sfp"
        plug_name = "sfp_tip"
        port_type = "sfp"
        port_name = "sfp_port_0"
        gripper_z = 0.04245

    return {
        "scene": {
            "task_board": task_board,
            "cables": {
                cable_name: {
                    "pose": {
                        "gripper_offset": {
                            "x": 0.0,
                            "y": 0.015385,
                            "z": gripper_z,
                        },
                        "roll": 0.4432,
                        "pitch": -0.4838,
                        "yaw": 1.3303,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": cable_type,
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": plug_type,
                "plug_name": plug_name,
                "port_type": port_type,
                "port_name": port_name,
                "target_module_name": scene_variant.target_name,
                "time_limit": 180,
            }
        },
    }


def write_engine_case_config(
    *,
    batch_root: Path,
    case_name: str,
    mode: str,
    scene_variant: SceneVariant,
) -> Path:
    payload = load_engine_template()
    payload["trials"] = {"trial_1": build_engine_trial(mode, scene_variant)}
    config_dir = ensure_dir(batch_root / "configs")
    config_path = config_dir / f"{case_name}.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def _sample_variant(
    mode: str,
    scene_profile: str,
    nic_count_profile: str,
    sfp_target_pool: str,
    rollout_profile: str,
    perturbation_profile: str,
    sc_demo_bucket_profile: str,
    sfp_demo_bucket_profile: str,
    case_index: int,
    rng: random.Random,
) -> SceneVariant:
    env: dict[str, str] = {
        # The sample qualification-style config varies task-board x/y/yaw while
        # keeping roll/pitch fixed at zero.
        "TASK_BOARD_X": _fmt(rng.uniform(0.15, 0.17)),
        "TASK_BOARD_Y": _fmt(rng.uniform(-0.2, 0.0)),
        "TASK_BOARD_Z": _fmt(1.14),
        "TASK_BOARD_ROLL": _fmt(0.0),
        "TASK_BOARD_PITCH": _fmt(0.0),
        "TASK_BOARD_YAW": _fmt(rng.uniform(3.0, 3.1415)),
        "SC_PORT_0_TRANSLATION": _fmt(rng.uniform(0.0, 0.115)),
        "SC_PORT_1_TRANSLATION": _fmt(rng.uniform(0.0, 0.115)),
    }

    # Qualification-focused generation only uses zones 1 and 2.
    for name in (
        "LC_MOUNT_RAIL_0",
        "LC_MOUNT_RAIL_1",
        "SFP_MOUNT_RAIL_0",
        "SFP_MOUNT_RAIL_1",
        "SC_MOUNT_RAIL_0",
        "SC_MOUNT_RAIL_1",
    ):
        env[f"{name}_PRESENT"] = "false"
        env[f"{name}_TRANSLATION"] = _fmt(0.0)
        env[f"{name}_ROLL"] = _fmt(0.0)
        env[f"{name}_PITCH"] = _fmt(0.0)
        env[f"{name}_YAW"] = _fmt(0.0)

    for name in ("SC_PORT_0", "SC_PORT_1"):
        # Keep SC ports aligned with the board for qualification-style
        # generation. The prose docs explicitly call out translation
        # randomization; roll/pitch are fixed at zero in evaluation docs.
        env[f"{name}_ROLL"] = _fmt(0.0)
        env[f"{name}_PITCH"] = _fmt(0.0)
        env[f"{name}_YAW"] = _fmt(0.0)
        env[f"{name}_PRESENT"] = "true"

    if mode == "sfp":
        if sfp_target_pool == "known_good":
            target_index = rng.choice((0, 1))
        else:
            target_index = rng.choice((0, 1, 2, 3, 4))
    else:
        target_index = rng.choice((0, 1))

    nic_present_indices = _sample_nic_present_indices(
        mode=mode,
        scene_profile=scene_profile,
        nic_count_profile=nic_count_profile,
        rng=rng,
        target_index=target_index if mode == "sfp" else None,
    )

    for idx in range(5):
        key = f"NIC_CARD_MOUNT_{idx}"
        env[f"{key}_TRANSLATION"] = _fmt(rng.uniform(0.0, 0.062))
        env[f"{key}_ROLL"] = _fmt(0.0)
        env[f"{key}_PITCH"] = _fmt(0.0)
        # NIC cards are documented with orientation offsets up to +/-10 deg.
        env[f"{key}_YAW"] = _fmt(rng.uniform(-0.0873, 0.0873))
        env[f"{key}_PRESENT"] = "true" if idx in nic_present_indices else "false"

    if scene_profile == "minimal":
        env["SFP_MOUNT_RAIL_1_PRESENT"] = "false"
        env["SC_MOUNT_RAIL_1_PRESENT"] = "false"
        env["SC_PORT_1_PRESENT"] = "false"
        for idx in range(1, 5):
            env[f"NIC_CARD_MOUNT_{idx}_PRESENT"] = "false"

    if mode == "sc":
        env["CABLE_TYPE"] = "sfp_sc_cable_reversed"
        env["SFP_MOUNT_RAIL_0_PRESENT"] = "false"
        if scene_profile == "minimal":
            env["NIC_CARD_MOUNT_0_PRESENT"] = "false"
        task_args = (
            "--task-id",
            "fixed_scene_sc",
            "--cable-type",
            "sfp_sc",
            "--cable-name",
            "cable_1",
            "--plug-type",
            "sc",
            "--plug-name",
            "sc_tip",
            "--port-type",
            "sc",
            "--port-name",
            "sc_port_base",
            "--target-module-name",
            f"sc_port_{target_index}",
            "--time-limit",
            "180",
        )
        target_name = f"sc_port_{target_index}"
    else:
        env["CABLE_TYPE"] = "sfp_sc_cable"
        env["SC_PORT_0_PRESENT"] = "true" if scene_profile == "generalization" else "false"
        env["SC_PORT_1_PRESENT"] = "true" if scene_profile == "generalization" else "false"
        task_args = (
            "--task-id",
            "fixed_scene_sfp",
            "--cable-type",
            "sfp_sc",
            "--cable-name",
            "cable_0",
            "--plug-type",
            "sfp",
            "--plug-name",
            "sfp_tip",
            "--port-type",
            "sfp",
            "--port-name",
            "sfp_port_0",
            "--target-module-name",
            f"nic_card_mount_{target_index}",
            "--time-limit",
            "180",
        )
        target_name = f"nic_card_mount_{target_index}"

    policy_env_overrides: dict[str, str] = {}
    resolved_perturbation_profile = "none"
    planned_demo_bucket = "clean_nominal_candidate"
    expected_primary_stage = "approach"
    recommended_collection_mode = "scripted"
    collection_notes = ""
    effective_rollout_profile = rollout_profile
    effective_perturbation_profile = perturbation_profile

    if mode == "sc" and sc_demo_bucket_profile != "default":
        bucket_spec = _planned_sc_demo_bucket_spec(
            case_index=case_index,
            rng=rng,
            profile=sc_demo_bucket_profile,
        )
        planned_demo_bucket = bucket_spec["planned_demo_bucket"]
        expected_primary_stage = bucket_spec["expected_primary_stage"]
        recommended_collection_mode = bucket_spec["recommended_collection_mode"]
        collection_notes = bucket_spec["collection_notes"]
        effective_rollout_profile = "perturb_sc"
        effective_perturbation_profile = bucket_spec["perturbation_profile"]
        policy_env_overrides.update(bucket_spec.get("policy_env_overrides", {}))

    if mode == "sfp" and sfp_demo_bucket_profile != "default":
        bucket_spec = _planned_sfp_demo_bucket_spec(
            case_index=case_index,
            rng=rng,
            profile=sfp_demo_bucket_profile,
        )
        planned_demo_bucket = bucket_spec["planned_demo_bucket"]
        expected_primary_stage = bucket_spec["expected_primary_stage"]
        recommended_collection_mode = bucket_spec["recommended_collection_mode"]
        collection_notes = bucket_spec["collection_notes"]
        effective_rollout_profile = "perturb_sfp"
        effective_perturbation_profile = bucket_spec["perturbation_profile"]
        policy_env_overrides.update(bucket_spec.get("policy_env_overrides", {}))

    if effective_rollout_profile in {"perturb_sc", "perturb_sfp"}:
        perturbation = _sample_sc_perturbation(effective_perturbation_profile, rng)
        policy_env_overrides.update(perturbation.env_overrides())
        resolved_perturbation_profile = perturbation.profile

    return SceneVariant(
        env_overrides=env,
        task_args=task_args,
        target_name=target_name,
        rollout_profile=effective_rollout_profile,
        perturbation_profile=resolved_perturbation_profile,
        policy_env_overrides=policy_env_overrides,
        planned_demo_bucket=planned_demo_bucket,
        expected_primary_stage=expected_primary_stage,
        recommended_collection_mode=recommended_collection_mode,
        collection_notes=collection_notes,
    )


def export_preview_video(
    *,
    dataset_root: Path,
    case_name: str,
    batch_root: Path,
    max_frames: int,
    scale: float,
) -> Path | None:
    if not dataset_has_episode_parquet(dataset_root):
        return None

    export_dir = ensure_dir(batch_root / "previews" / case_name)
    command = [
        "bash",
        str(REPO_ROOT / "quantarune_training" / "bash" / "peek_dataset.sh"),
        str(dataset_root),
        "--mode",
        "export",
        "--export-dir",
        str(export_dir),
        "--max-frames",
        str(max_frames),
        "--scale",
        str(scale),
        "--no-open",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=False)

    videos = sorted(export_dir.glob("*.mp4"))
    return videos[0] if videos else None


def load_case_scoring(
    dataset_root: Path,
    fixed_scene_root: Path,
) -> tuple[Path | None, dict | None]:
    candidate_paths = (
        dataset_root / "scoring.yaml",
        fixed_scene_root / "scoring.yaml",
    )
    for path in candidate_paths:
        payload = load_yaml(path)
        if isinstance(payload, dict) and "total" in payload:
            return path, payload
    return None, None


def load_teacher_stage_trace(
    dataset_root: Path,
    fixed_scene_root: Path,
) -> tuple[Path | None, dict | None]:
    candidate_paths = (
        dataset_root / "teacher_stage_trace.json",
        fixed_scene_root / "teacher_stage_trace.json",
    )
    for path in candidate_paths:
        payload = load_json(path)
        if isinstance(payload, dict) and payload.get("schema") and payload.get("events"):
            return path, payload
    return None, None


def run_case(
    *,
    case_name: str,
    mode_spec: ModeSpec,
    scene_variant: SceneVariant,
    batch_root: Path,
    record: bool,
    skip_audit: bool,
    engine_wait: float,
    policy_wait: float,
    keep_running: bool,
    audit_root: Path,
    scene_profile: str,
    policy_import: str | None,
    policy_mode: str = "full",
    handoff_z_offset_m: float | None = None,
    handoff_settle_s: float | None = None,
    result_timeout_seconds: float | None = None,
) -> dict[str, object]:
    config_path = write_engine_case_config(
        batch_root=audit_root.parent,
        case_name=case_name,
        mode=mode_spec.mode,
        scene_variant=scene_variant,
    )
    fixed_scene_root = DATA_ROOT / "outputs" / "fixed_scene_runs" / case_name
    runtime_geometry_path = (
        batch_root / "runtime_geometry" / case_name / "runtime_lateral_correction.jsonl"
    )

    command = [
        "bash",
        str(DATA_ROOT / "bash" / "run_engine_case_cheatcode.sh"),
        "--config-file",
        str(config_path),
        "--session-name",
        case_name,
        "--record-session",
        case_name,
        "--record-trials",
        "1",
        "--policy-mode",
        policy_mode,
        "--engine-wait",
        str(engine_wait),
        "--policy-wait",
        str(policy_wait),
    ]

    if policy_import:
        command.extend(["--policy-import", policy_import])
    if handoff_z_offset_m is not None:
        command.extend(["--handoff-z-offset-m", str(handoff_z_offset_m)])
    if handoff_settle_s is not None:
        command.extend(["--handoff-settle-s", str(handoff_settle_s)])
    if result_timeout_seconds is not None:
        command.extend(["--result-timeout", str(result_timeout_seconds)])

    env = os.environ.copy()
    env.update(scene_variant.policy_env_overrides or {})
    env.setdefault("AIC_CHEATCODE_RUNTIME_GEOMETRY_JSONL", str(runtime_geometry_path))
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)

    task_result_path = fixed_scene_root / "task_result.json"
    task_result = load_json(task_result_path) or {}

    recorded_session_root = DATA_ROOT / "outputs" / "gazebo_data" / case_name
    dataset_root = recorded_session_root / "parquet_root" / "experiment_1"
    audit_dir = audit_root / case_name
    audit_summary = None
    dataset_has_data = False
    dataset_has_readable_data = False

    if record:
        dataset_has_data, dataset_has_readable_data = wait_for_dataset_materialization(
            dataset_root
        )
    else:
        dataset_has_data = dataset_has_episode_parquet(dataset_root)
        dataset_has_readable_data = (
            dataset_has_data and dataset_has_readable_parquet(dataset_root)
        )

    if record:
        recorded_session_root, dataset_root = relocate_recorded_session(
            case_name=case_name,
            batch_root=batch_root,
        )

    if record and not skip_audit and dataset_has_readable_data:
        audit_command = [
            "bash",
            str(REPO_ROOT / "quantarune_training" / "bash" / "audit_dataset.sh"),
            str(dataset_root),
            "--output-dir",
            str(audit_dir),
            "--force",
        ]
        subprocess.run(audit_command, cwd=REPO_ROOT, check=False)
        audit_summary = load_json(audit_dir / "summary.json")

    bucket = None
    if audit_summary is not None:
        bucket_counts = audit_summary.get("bucket_counts", {})
        if bucket_counts:
            bucket = max(bucket_counts, key=bucket_counts.get)
    elif record and dataset_has_data and not dataset_has_readable_data:
        bucket = "corrupt_dataset"
    elif record and not dataset_has_data:
        bucket = "empty_dataset"

    scoring_path, scoring_payload = load_case_scoring(dataset_root, fixed_scene_root)
    stage_trace_path, stage_trace_payload = load_teacher_stage_trace(
        dataset_root,
        fixed_scene_root,
    )
    trial_payload = {}
    if isinstance(scoring_payload, dict):
        trial_payload = scoring_payload.get("trial_1", {}) or {}

    return {
        "case_name": case_name,
        "mode": mode_spec.mode,
        "rollout_profile": scene_variant.rollout_profile,
        "perturbation_profile": scene_variant.perturbation_profile,
        "launcher_exit_code": int(completed.returncode),
        "engine_config_path": str(config_path),
        "task_result_json": str(task_result_path),
        "task_exit_code": task_result.get("exit_code"),
        "goal_accepted": task_result.get("goal_accepted"),
        "goal_success": task_result.get("goal_success"),
        "goal_status": task_result.get("goal_status"),
        "goal_message": task_result.get("goal_message"),
        "fixed_scene_root": str(fixed_scene_root),
        "recorded_session_root": str(recorded_session_root) if recorded_session_root.exists() else "",
        "dataset_root": str(dataset_root) if dataset_root.exists() else "",
        "dataset_has_data": dataset_has_data,
        "dataset_has_readable_data": dataset_has_readable_data,
        "scoring_yaml": str(scoring_path) if scoring_path else "",
        "score_total": scoring_payload.get("total") if scoring_payload else "",
        "score_tier_1": ((trial_payload.get("tier_1") or {}).get("score", "")),
        "score_tier_2": ((trial_payload.get("tier_2") or {}).get("score", "")),
        "score_tier_3": ((trial_payload.get("tier_3") or {}).get("score", "")),
        "score_tier_1_message": ((trial_payload.get("tier_1") or {}).get("message", "")),
        "score_tier_2_message": ((trial_payload.get("tier_2") or {}).get("message", "")),
        "score_tier_3_message": ((trial_payload.get("tier_3") or {}).get("message", "")),
        "audit_dir": str(audit_dir) if audit_dir.exists() else "",
        "bucket": bucket or "",
        "scene_profile": scene_profile,
        "policy_import": policy_import or "",
        "target_name": scene_variant.target_name,
        "policy_env": json.dumps(scene_variant.policy_env_overrides or {}, sort_keys=True),
        "runtime_geometry_stream": (
            str(runtime_geometry_path) if runtime_geometry_path.exists() else ""
        ),
        "teacher_stage_trace": str(stage_trace_path) if stage_trace_path else "",
        "teacher_stage_schema": (
            str(stage_trace_payload.get("schema"))
            if isinstance(stage_trace_payload, dict)
            else ""
        ),
        "teacher_stage_names": (
            json.dumps(stage_trace_payload.get("stage_names", []))
            if isinstance(stage_trace_payload, dict)
            else "[]"
        ),
        "teacher_stage_event_count": (
            len(stage_trace_payload.get("events", []))
            if isinstance(stage_trace_payload, dict)
            else 0
        ),
        "planned_demo_bucket": scene_variant.planned_demo_bucket,
        "expected_primary_stage": scene_variant.expected_primary_stage,
        "recommended_collection_mode": scene_variant.recommended_collection_mode,
        "collection_notes": scene_variant.collection_notes,
        "perturb_x_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_X_M",
            "",
        ),
        "perturb_y_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_Y_M",
            "",
        ),
        "perturb_z_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_Z_M",
            "",
        ),
        "perturb_roll_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_ROLL_RAD",
            "",
        ),
        "perturb_pitch_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_PITCH_RAD",
            "",
        ),
        "perturb_yaw_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_YAW_RAD",
            "",
        ),
        "perturb_decay_end_z_offset_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_DECAY_END_Z_OFFSET_M",
            "",
        ),
        "nic_present_count": sum(
            1
            for idx in range(5)
            if scene_variant.env_overrides.get(f"NIC_CARD_MOUNT_{idx}_PRESENT") == "true"
        ),
        "task_args": " ".join(scene_variant.task_args),
        "scene_env": json.dumps(scene_variant.env_overrides, sort_keys=True),
    }


def plan_case(
    *,
    case_name: str,
    mode_spec: ModeSpec,
    scene_variant: SceneVariant,
    batch_root: Path,
    scene_profile: str,
    policy_import: str | None,
) -> dict[str, object]:
    config_path = write_engine_case_config(
        batch_root=batch_root,
        case_name=case_name,
        mode=mode_spec.mode,
        scene_variant=scene_variant,
    )
    fixed_scene_root = batch_root / "planned_cases" / case_name
    ensure_dir(fixed_scene_root)

    row = {
        "case_name": case_name,
        "mode": mode_spec.mode,
        "rollout_profile": scene_variant.rollout_profile,
        "perturbation_profile": scene_variant.perturbation_profile,
        "launcher_exit_code": "",
        "engine_config_path": str(config_path),
        "task_result_json": "",
        "task_exit_code": "",
        "goal_accepted": "",
        "goal_success": "",
        "goal_status": "",
        "goal_message": "",
        "fixed_scene_root": str(fixed_scene_root),
        "recorded_session_root": "",
        "dataset_root": "",
        "dataset_has_data": False,
        "dataset_has_readable_data": False,
        "scoring_yaml": "",
        "score_total": "",
        "score_tier_1": "",
        "score_tier_2": "",
        "score_tier_3": "",
        "score_tier_1_message": "",
        "score_tier_2_message": "",
        "score_tier_3_message": "",
        "audit_dir": "",
        "bucket": "",
        "scene_profile": scene_profile,
        "policy_import": policy_import or "",
        "target_name": scene_variant.target_name,
        "policy_env": json.dumps(scene_variant.policy_env_overrides or {}, sort_keys=True),
        "teacher_stage_trace": "",
        "teacher_stage_schema": "",
        "teacher_stage_names": "[]",
        "teacher_stage_event_count": 0,
        "planned_demo_bucket": scene_variant.planned_demo_bucket,
        "expected_primary_stage": scene_variant.expected_primary_stage,
        "recommended_collection_mode": scene_variant.recommended_collection_mode,
        "collection_notes": scene_variant.collection_notes,
        "perturb_x_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_X_M",
            "",
        ),
        "perturb_y_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_Y_M",
            "",
        ),
        "perturb_z_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_Z_M",
            "",
        ),
        "perturb_roll_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_ROLL_RAD",
            "",
        ),
        "perturb_pitch_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_PITCH_RAD",
            "",
        ),
        "perturb_yaw_rad": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_YAW_RAD",
            "",
        ),
        "perturb_decay_end_z_offset_m": (scene_variant.policy_env_overrides or {}).get(
            "AIC_CHEATCODE_PERTURB_DECAY_END_Z_OFFSET_M",
            "",
        ),
        "nic_present_count": sum(
            1
            for idx in range(5)
            if scene_variant.env_overrides.get(f"NIC_CARD_MOUNT_{idx}_PRESENT") == "true"
        ),
        "task_args": " ".join(scene_variant.task_args),
        "scene_env": json.dumps(scene_variant.env_overrides, sort_keys=True),
        "generation_mode": "manifest_only",
        "recording_ready": True,
    }
    (fixed_scene_root / "case_manifest.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def write_case_manifest(row: dict[str, object]) -> None:
    fieldnames = list(row.keys())
    output_roots: list[Path] = []

    fixed_scene_root = Path(str(row["fixed_scene_root"])).expanduser()
    output_roots.append(fixed_scene_root)

    dataset_root_value = str(row.get("dataset_root") or "").strip()
    if dataset_root_value:
        output_roots.append(Path(dataset_root_value).expanduser())

    for output_root in output_roots:
        if not output_root.exists():
            continue
        with (output_root / "cases.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)


def write_outputs(
    batch_root: Path,
    rows: list[dict[str, object]],
    *,
    mode: str,
    scene_profile: str,
    nic_count_profile: str,
    sfp_target_pool: str,
    seed: int,
    rollout_profile: str,
    perturbation_profile: str,
    sc_demo_bucket_profile: str,
    sfp_demo_bucket_profile: str,
) -> None:
    summary_path = batch_root / "cases.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    counter = Counter()
    for row in rows:
        bucket = str(
            row.get("bucket")
            or row.get("planned_demo_bucket")
            or "unbucketed"
        )
        counter[bucket] += 1

    payload = {
        "created_at": datetime.now().isoformat(),
        "mode": mode,
        "scene_profile": scene_profile,
        "nic_count_profile": nic_count_profile,
        "sfp_target_pool": sfp_target_pool,
        "seed": seed,
        "rollout_profile": rollout_profile,
        "perturbation_profile": perturbation_profile,
        "sc_demo_bucket_profile": sc_demo_bucket_profile,
        "sfp_demo_bucket_profile": sfp_demo_bucket_profile,
        "generation_mode": (
            rows[-1].get("generation_mode", "executed") if rows else "executed"
        ),
        "cases_total": len(rows),
        "bucket_counts": dict(counter),
        "rows": rows,
    }
    (batch_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    resolved_seed = _resolve_seed(args.seed)
    mode_spec = MODE_SPECS[args.mode]
    prefix = args.session_prefix or f"fixed_scene_{args.mode}_{timestamp()}"
    batch_root = ensure_dir(OUTPUT_ROOT / prefix)
    audit_root = ensure_dir(batch_root / "audits")
    rng = random.Random(resolved_seed)

    rows: list[dict[str, object]] = []
    for index in range(1, args.count + 1):
        case_name = f"{prefix}_{index:03d}"
        scene_variant = _sample_variant(
            args.mode,
            args.scene_profile,
            args.nic_count_profile,
            args.sfp_target_pool,
            args.rollout_profile,
            args.perturbation_profile,
            args.sc_demo_bucket_profile,
            args.sfp_demo_bucket_profile,
            index,
            rng,
        )
        if args.manifest_only:
            row = plan_case(
                case_name=case_name,
                mode_spec=mode_spec,
                scene_variant=scene_variant,
                batch_root=batch_root,
                scene_profile=args.scene_profile,
                policy_import=args.policy_import,
            )
        else:
            row = run_case(
                case_name=case_name,
                mode_spec=mode_spec,
                scene_variant=scene_variant,
                batch_root=batch_root,
                record=not args.no_record,
                skip_audit=args.skip_audit,
                engine_wait=args.engine_wait,
                policy_wait=args.policy_wait,
                keep_running=args.keep_case_stacks,
                audit_root=audit_root,
                scene_profile=args.scene_profile,
                policy_import=args.policy_import,
            )
        if args.export_preview and row["dataset_root"]:
            preview_path = export_preview_video(
                dataset_root=Path(str(row["dataset_root"])),
                case_name=case_name,
                batch_root=batch_root,
                max_frames=args.preview_max_frames,
                scale=args.preview_scale,
            )
            row["preview_video"] = str(preview_path) if preview_path else ""
        row["seed"] = resolved_seed
        row["nic_count_profile"] = args.nic_count_profile
        row["sfp_target_pool"] = args.sfp_target_pool
        row["sc_demo_bucket_profile"] = args.sc_demo_bucket_profile
        row["sfp_demo_bucket_profile"] = args.sfp_demo_bucket_profile
        write_case_manifest(row)
        rows.append(row)
        write_outputs(
            batch_root,
            rows,
            mode=args.mode,
            scene_profile=args.scene_profile,
            nic_count_profile=args.nic_count_profile,
            sfp_target_pool=args.sfp_target_pool,
            seed=resolved_seed,
            rollout_profile=args.rollout_profile,
            perturbation_profile=args.perturbation_profile,
            sc_demo_bucket_profile=args.sc_demo_bucket_profile,
            sfp_demo_bucket_profile=args.sfp_demo_bucket_profile,
        )

    counter = Counter(
        str(row.get("bucket") or row.get("planned_demo_bucket") or "unbucketed")
        for row in rows
    )
    print(f"Batch written to: {batch_root}")
    print("Bucket counts:")
    for bucket_name, count in sorted(counter.items()):
        print(f"  - {bucket_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
