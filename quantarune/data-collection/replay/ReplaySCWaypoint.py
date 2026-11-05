#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from std_msgs.msg import String


@dataclass(frozen=True)
class ReplayWaypoint:
    timestamp_s: float
    pose: Pose


@dataclass(frozen=True)
class ReplayTemplateCandidate:
    name: str
    path: Path
    source: str
    priority: float = 0.0
    target_module_name: str = ""
    port_type: str = ""
    plug_type: str = ""
    target_port_id: str = ""
    target_module_id: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _policy_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_template_path() -> Path:
    packaged = (
        Path(__file__).resolve().parent
        / "replay_templates"
        / "sc_scripted_full_001_026_actions.jsonl"
    )
    if packaged.exists():
        return packaged
    return (
        _repo_root()
        / "tools"
        / "insertion_harness"
        / "examples"
        / "real_episodes"
        / "phase2_scripted_fixed_scene"
        / "sc_scripted_full_001_026"
        / "observations"
        / "actions.jsonl"
    )


class RunSCWaypointReplay(Policy):
    """Replay a recorded successful SC TCP trajectory without scoring TF.

    This is intentionally narrow: it tests whether fixed-scene scripted cases can be
    converted into a deployable policy lane. It does not read ground-truth port or
    plug transforms at runtime.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.template_manifest_path = self._optional_path_env(
            "AIC_SC_REPLAY_TEMPLATE_MANIFEST"
        )
        self.template_library_dir = self._optional_path_env("AIC_SC_REPLAY_TEMPLATE_DIR")
        self.template_selection = os.environ.get(
            "AIC_SC_REPLAY_TEMPLATE_SELECTION", "task"
        ).strip().lower()
        self.template_id_override = os.environ.get(
            "AIC_SC_REPLAY_TEMPLATE_ID", ""
        ).strip().lower()
        self.template_candidates = self._discover_template_candidates()
        self.template_candidate = self._select_template_candidate(None)
        self.template_path = self.template_candidate.path
        self.template_name = self.template_candidate.name
        self.template_metadata = dict(self.template_candidate.metadata)

        self.stride = max(1, _int_env("AIC_SC_REPLAY_STRIDE", 1))
        self.max_waypoints = _int_env("AIC_SC_REPLAY_MAX_WAYPOINTS", 0)
        self.period_s = max(0.01, _float_env("AIC_SC_REPLAY_PERIOD_S", 0.08))
        self.final_hold_s = max(0.0, _float_env("AIC_SC_REPLAY_FINAL_HOLD_S", 5.0))
        self.x_offset_m = _float_env("AIC_SC_REPLAY_X_OFFSET_M", 0.0)
        self.y_offset_m = _float_env("AIC_SC_REPLAY_Y_OFFSET_M", 0.0)
        self.z_offset_m = _float_env("AIC_SC_REPLAY_Z_OFFSET_M", 0.0)
        self.final_z_nudge_m = _float_env("AIC_SC_REPLAY_FINAL_Z_NUDGE_M", 0.0)
        self.final_nudge_steps = max(0, _int_env("AIC_SC_REPLAY_FINAL_NUDGE_STEPS", 0))
        self.stop_on_insertion_event = (
            os.environ.get("AIC_SC_REPLAY_STOP_ON_INSERTION_EVENT", "1").lower()
            not in {"0", "false", "no"}
        )
        self.soft_force_n = _float_env("AIC_SC_REPLAY_SOFT_FORCE_N", 32.0)
        self.hard_force_n = _float_env("AIC_SC_REPLAY_HARD_FORCE_N", 45.0)
        self.hard_force_consecutive_steps = max(
            1, _int_env("AIC_SC_REPLAY_HARD_FORCE_CONSECUTIVE_STEPS", 3)
        )
        self.backout_waypoints = max(0, _int_env("AIC_SC_REPLAY_BACKOUT_WAYPOINTS", 24))
        self.stiffness = self._diag_env(
            "AIC_SC_REPLAY_STIFFNESS_DIAG", [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        )
        self.damping = self._diag_env(
            "AIC_SC_REPLAY_DAMPING_DIAG", [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
        )

        self.insertion_events: list[str] = []
        self.last_insertion_event = ""
        self.last_force_magnitude_n = 0.0
        self.max_force_magnitude_n = 0.0

        self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_cb,
            10,
        )
        self._parent_node.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._wrench_cb,
            10,
        )

        self.waypoints = self._load_waypoints(self.template_path)
        self.get_logger().info(
            "RunSCWaypointReplay configured: "
            f"template={self.template_name} path={self.template_path} "
            f"source={self.template_candidate.source} waypoints={len(self.waypoints)} "
            f"stride={self.stride} period_s={self.period_s:.3f} "
            f"offset=({self.x_offset_m:.4f},{self.y_offset_m:.4f},{self.z_offset_m:.4f}) "
            f"final_z_nudge_m={self.final_z_nudge_m:.4f} "
            f"soft_force_n={self.soft_force_n:.1f} hard_force_n={self.hard_force_n:.1f}"
        )

    @staticmethod
    def _diag_env(name: str, default: list[float]) -> list[float]:
        raw = os.environ.get(name)
        if not raw:
            return default
        try:
            values = [float(part.strip()) for part in raw.split(",")]
        except ValueError:
            return default
        return values if len(values) == 6 else default

    def _insertion_event_cb(self, msg: String) -> None:
        self.last_insertion_event = msg.data
        self.insertion_events.append(msg.data)
        self.get_logger().info(f"SC replay insertion_event: {msg.data}")

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        force = msg.wrench.force
        magnitude = math.sqrt(force.x * force.x + force.y * force.y + force.z * force.z)
        self.last_force_magnitude_n = magnitude
        self.max_force_magnitude_n = max(self.max_force_magnitude_n, magnitude)

    @staticmethod
    def _optional_path_env(name: str) -> Path | None:
        value = os.environ.get(name, "").strip()
        return Path(value).expanduser() if value else None

    @staticmethod
    def _metadata_str(metadata: dict[str, object], key: str) -> str:
        value = metadata.get(key, "")
        return str(value).strip().lower() if value is not None else ""

    @staticmethod
    def _task_field(task: Task | None, field_name: str) -> str:
        if task is None or not hasattr(task, field_name):
            return ""
        value = getattr(task, field_name)
        return str(value).strip().lower() if value is not None else ""

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_replay_template_path(self, path: Path) -> Path:
        path = path.expanduser()
        if path.is_absolute() and not path.exists() and "aic-quantarune" in path.parts:
            repo_index = path.parts.index("aic-quantarune")
            remapped = _policy_repo_root().joinpath(*path.parts[repo_index + 1 :])
            if remapped.exists():
                path = remapped
        if path.is_absolute() and not path.exists() and "route_libraries" in path.parts:
            manifest_path = self.template_manifest_path
            if manifest_path is not None and manifest_path.exists():
                route_index = path.parts.index("route_libraries")
                route_parts = path.parts[route_index + 1 :]
                if route_parts:
                    if route_parts[0] == manifest_path.parent.name:
                        remapped = manifest_path.parent.joinpath(*route_parts[1:])
                    else:
                        remapped = manifest_path.parent.parent.joinpath(*route_parts)
                    if remapped.exists():
                        path = remapped
        if not path.is_dir():
            return path
        for candidate in (
            path / "observations" / "actions.jsonl",
            path / "actions.jsonl",
            path / "observations" / "robot_state.jsonl",
            path / "robot_state.jsonl",
        ):
            if candidate.exists():
                return candidate
        return path

    def _make_template_candidate(
        self,
        *,
        path: Path,
        name: str = "",
        source: str,
        metadata: dict[str, object] | None = None,
        base_dir: Path | None = None,
    ) -> ReplayTemplateCandidate:
        metadata = dict(metadata or {})
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        resolved_path = self._resolve_replay_template_path(path)
        candidate_name = name or str(metadata.get("name") or resolved_path.stem)
        return ReplayTemplateCandidate(
            name=candidate_name,
            path=resolved_path,
            source=source,
            priority=self._safe_float(metadata.get("priority"), 0.0),
            target_module_name=self._metadata_str(metadata, "target_module_name"),
            port_type=self._metadata_str(metadata, "port_type"),
            plug_type=self._metadata_str(metadata, "plug_type"),
            target_port_id=self._metadata_str(metadata, "target_port_id"),
            target_module_id=self._metadata_str(metadata, "target_module_id"),
            metadata=metadata,
        )

    def _manifest_template_candidates(self) -> list[ReplayTemplateCandidate]:
        manifest_path = self.template_manifest_path
        if manifest_path is None or not manifest_path.exists():
            return []
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            entries = payload.get("templates", [])
        else:
            entries = payload
        if not isinstance(entries, list):
            self.get_logger().warning(
                f"SC replay template manifest ignored; templates is not a list: {manifest_path}"
            )
            return []

        candidates: list[ReplayTemplateCandidate] = []
        for index, entry in enumerate(entries):
            metadata: dict[str, object]
            if isinstance(entry, str):
                path_value = entry
                metadata = {"name": Path(entry).stem}
            elif isinstance(entry, dict):
                metadata = dict(entry)
                path_value = (
                    entry.get("path")
                    or entry.get("template_path")
                    or entry.get("actions_path")
                    or entry.get("output_actions_jsonl")
                    or entry.get("episode_dir")
                )
            else:
                self.get_logger().warning(
                    f"SC replay template manifest entry {index} ignored; unsupported type"
                )
                continue
            if not path_value:
                self.get_logger().warning(
                    f"SC replay template manifest entry {index} ignored; missing path"
                )
                continue
            candidates.append(
                self._make_template_candidate(
                    path=Path(str(path_value)),
                    name=str(metadata.get("name") or ""),
                    source="manifest",
                    metadata=metadata,
                    base_dir=manifest_path.parent,
                )
            )
        return candidates

    def _directory_template_candidates(self) -> list[ReplayTemplateCandidate]:
        root = self.template_library_dir
        if root is None or not root.exists():
            return []
        paths = set(root.rglob("actions.jsonl"))
        paths.update(root.rglob("*_actions.jsonl"))
        paths.update(root.rglob("robot_state.jsonl"))
        candidates: list[ReplayTemplateCandidate] = []
        for path in sorted(paths):
            if "__pycache__" in path.parts:
                continue
            if path.name == "robot_state.jsonl" and path.with_name("actions.jsonl").exists():
                continue
            episode_dir = path.parent.parent if path.parent.name == "observations" else path.parent
            name = episode_dir.name if episode_dir != root else path.stem
            candidates.append(
                self._make_template_candidate(
                    path=path,
                    name=name,
                    source="directory",
                    metadata={"name": name},
                )
            )
        return candidates

    def _discover_template_candidates(self) -> list[ReplayTemplateCandidate]:
        default_path = Path(
            os.environ.get("AIC_SC_REPLAY_EPISODE_DIR", str(_default_template_path()))
        )
        candidates = [
            self._make_template_candidate(
                path=default_path,
                name=os.environ.get("AIC_SC_REPLAY_TEMPLATE_NAME", "default"),
                source="default",
                metadata={"name": os.environ.get("AIC_SC_REPLAY_TEMPLATE_NAME", "default")},
            )
        ]
        candidates.extend(self._manifest_template_candidates())
        candidates.extend(self._directory_template_candidates())

        deduped: dict[str, ReplayTemplateCandidate] = {}
        for candidate in candidates:
            key = str(candidate.path)
            if key not in deduped:
                deduped[key] = candidate
        return list(deduped.values())

    def _score_template_candidate(
        self, candidate: ReplayTemplateCandidate, task: Task | None
    ) -> float:
        score = candidate.priority
        for candidate_field, task_field in (
            ("target_module_name", "target_module_name"),
            ("port_type", "port_type"),
            ("plug_type", "plug_type"),
            ("target_port_id", "target_port_id"),
            ("target_module_id", "target_module_id"),
        ):
            expected = getattr(candidate, candidate_field)
            if not expected:
                continue
            actual = self._task_field(task, task_field)
            score += 100.0 if actual == expected else -50.0
        return score

    def _select_template_candidate(self, task: Task | None) -> ReplayTemplateCandidate:
        if not self.template_candidates:
            raise FileNotFoundError("No SC replay template candidates were discovered")
        if self.template_id_override:
            for candidate in self.template_candidates:
                names = {
                    candidate.name.lower(),
                    candidate.path.stem.lower(),
                    str(candidate.path).lower(),
                }
                if self.template_id_override in names:
                    return candidate
            self.get_logger().warning(
                "AIC_SC_REPLAY_TEMPLATE_ID did not match any candidate; "
                f"requested={self.template_id_override!r} "
                f"available={[candidate.name for candidate in self.template_candidates]}"
            )
        if self.template_selection in {"first", "default", "off"}:
            return self.template_candidates[0]
        return max(
            self.template_candidates,
            key=lambda candidate: self._score_template_candidate(candidate, task),
        )

    def _select_replay_template_for_task(self, task: Task | None) -> None:
        candidate = self._select_template_candidate(task)
        if candidate.path == self.template_path:
            return
        self.template_candidate = candidate
        self.template_path = candidate.path
        self.template_name = candidate.name
        self.template_metadata = dict(candidate.metadata)
        self.waypoints = self._load_waypoints(self.template_path)
        self.get_logger().info(
            "SC replay template selected: "
            f"name={self.template_name} source={candidate.source} "
            f"path={self.template_path} waypoints={len(self.waypoints)} "
            f"task=({self._task_summary(task) if task is not None else 'none'})"
        )

    def _load_waypoints(self, path: Path) -> list[ReplayWaypoint]:
        if not path.exists():
            raise FileNotFoundError(f"SC replay template not found: {path}")

        waypoints: list[ReplayWaypoint] = []
        with path.open("r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                if index % self.stride != 0:
                    continue
                record = json.loads(line)
                position = record.get("absolute_pose_position")
                quat_xyzw = record.get("absolute_pose_quaternion")
                if not isinstance(position, list) or not isinstance(quat_xyzw, list):
                    position = record.get("end_effector_position")
                    quat_xyzw = record.get("end_effector_quaternion")
                if not isinstance(position, list) or not isinstance(quat_xyzw, list):
                    continue
                if len(position) != 3 or len(quat_xyzw) != 4:
                    continue
                if (
                    abs(float(position[0])) < 1e-9
                    and abs(float(position[1])) < 1e-9
                    and abs(float(position[2])) < 1e-9
                    and abs(float(quat_xyzw[0])) < 1e-9
                    and abs(float(quat_xyzw[1])) < 1e-9
                    and abs(float(quat_xyzw[2])) < 1e-9
                    and abs(float(quat_xyzw[3])) < 1e-9
                ):
                    continue
                pose = Pose(
                    position=Point(
                        x=float(position[0]) + self.x_offset_m,
                        y=float(position[1]) + self.y_offset_m,
                        z=float(position[2]) + self.z_offset_m,
                    ),
                    orientation=Quaternion(
                        x=float(quat_xyzw[0]),
                        y=float(quat_xyzw[1]),
                        z=float(quat_xyzw[2]),
                        w=float(quat_xyzw[3]),
                    ),
                )
                waypoints.append(
                    ReplayWaypoint(
                        timestamp_s=float(record.get("timestamp", 0.0)),
                        pose=pose,
                    )
                )
                if self.max_waypoints > 0 and len(waypoints) >= self.max_waypoints:
                    break

        if not waypoints:
            raise ValueError(f"No usable SC replay waypoints in {path}")
        return waypoints

    @staticmethod
    def _task_summary(task: Task) -> str:
        fields = [
            "cable_name",
            "plug_name",
            "port_name",
            "target_module_name",
            "target_module_id",
            "target_port_id",
        ]
        parts = []
        for field in fields:
            if hasattr(task, field):
                parts.append(f"{field}={getattr(task, field)}")
        return ", ".join(parts)

    def _hold_pose(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        hold_s: float,
        send_feedback: SendFeedbackCallback,
        reason: str,
    ) -> None:
        deadline = time.time() + hold_s
        while time.time() < deadline:
            self.set_pose_target(
                move_robot=move_robot,
                pose=pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            send_feedback(reason)
            self.sleep_for(min(self.period_s, 0.1))

    @staticmethod
    def _offset_pose_z(pose: Pose, z_offset_m: float) -> Pose:
        return Pose(
            position=Point(
                x=pose.position.x,
                y=pose.position.y,
                z=pose.position.z + z_offset_m,
            ),
            orientation=Quaternion(
                x=pose.orientation.x,
                y=pose.orientation.y,
                z=pose.orientation.z,
                w=pose.orientation.w,
            ),
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"RunSCWaypointReplay.insert_cable() task: {self._task_summary(task)}"
        )
        self._select_replay_template_for_task(task)

        hard_force_steps = 0
        last_pose = self.waypoints[0].pose
        start_time = time.time()

        for step, waypoint in enumerate(self.waypoints):
            loop_start = time.time()
            observation = get_observation()
            if observation is not None:
                wrench = observation.controller_state.fts_tare_offset.wrench.force
                obs_force = math.sqrt(
                    wrench.x * wrench.x + wrench.y * wrench.y + wrench.z * wrench.z
                )
                self.max_force_magnitude_n = max(self.max_force_magnitude_n, obs_force)

            force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)
            if force_n >= self.soft_force_n and step % 5 == 0:
                self.get_logger().warning(
                    f"SC replay soft force warning step={step} force_n={force_n:.2f}"
                )

            if force_n >= self.hard_force_n:
                hard_force_steps += 1
            else:
                hard_force_steps = 0

            if hard_force_steps >= self.hard_force_consecutive_steps:
                backout_index = max(0, step - self.backout_waypoints)
                backout_pose = self.waypoints[backout_index].pose
                self.get_logger().error(
                    "SC replay hard force abort: "
                    f"step={step} force_n={force_n:.2f} "
                    f"backout_index={backout_index}"
                )
                self._hold_pose(
                    move_robot=move_robot,
                    pose=backout_pose,
                    hold_s=0.75,
                    send_feedback=send_feedback,
                    reason="hard force backout",
                )
                break

            self.set_pose_target(
                move_robot=move_robot,
                pose=waypoint.pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            last_pose = waypoint.pose

            if step % 20 == 0:
                self.get_logger().info(
                    "SC replay step="
                    f"{step}/{len(self.waypoints)} "
                    f"template_t={waypoint.timestamp_s:.2f}s "
                    f"force_n={force_n:.2f} "
                    f"insertion_events={len(self.insertion_events)}"
                )
            send_feedback(
                f"SC replay step={step} force={force_n:.2f} events={len(self.insertion_events)}"
            )

            if self.stop_on_insertion_event and self.insertion_events:
                self.get_logger().info(
                    "SC replay stopping on insertion event: "
                    f"{self.last_insertion_event}"
                )
                break

            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.period_s - elapsed))

        if self.final_nudge_steps > 0 and self.final_z_nudge_m != 0.0:
            self.get_logger().info(
                "SC replay terminal z-nudge: "
                f"steps={self.final_nudge_steps} total_z={self.final_z_nudge_m:.4f}m"
            )
            for nudge_step in range(1, self.final_nudge_steps + 1):
                if self.stop_on_insertion_event and self.insertion_events:
                    break
                force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)
                if force_n >= self.hard_force_n:
                    self.get_logger().error(
                        "SC replay terminal z-nudge hard force abort: "
                        f"step={nudge_step} force_n={force_n:.2f}"
                    )
                    break
                fraction = nudge_step / float(self.final_nudge_steps)
                last_pose = self._offset_pose_z(last_pose, self.final_z_nudge_m / self.final_nudge_steps)
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=last_pose,
                    stiffness=self.stiffness,
                    damping=self.damping,
                )
                send_feedback(
                    f"SC replay terminal z-nudge={fraction:.2f} force={force_n:.2f}"
                )
                self.sleep_for(self.period_s)

        if self.final_hold_s > 0.0:
            self._hold_pose(
                move_robot=move_robot,
                pose=last_pose,
                hold_s=self.final_hold_s,
                send_feedback=send_feedback,
                reason="final hold",
            )

        self.get_logger().info(
            "RunSCWaypointReplay exiting: "
            f"events={self.insertion_events} "
            f"max_force_n={self.max_force_magnitude_n:.2f} "
            f"duration_s={time.time() - start_time:.2f}"
        )
        return True
