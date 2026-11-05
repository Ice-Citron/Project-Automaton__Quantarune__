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

import os
import time

import numpy as np

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .RunSCWaypointReplay import _float_env, _int_env
from .RunSCVisualServoReplay import RunSCVisualServoReplay


class RunSCAnchorGentleCommit(RunSCVisualServoReplay):
    """Replay to a fixed SC anchor, then run a force-guarded gentle commit.

    This is intentionally narrower than `RunSCVisualServoReplay`: it does not try
    heuristic visual centering. It is a diagnostic/control probe for the two
    recovered handoff states that currently matter:

    - `001_010@310`: main pre-insert local-correction handoff
    - `001_010@320`: late-stage diagnostic / reseat-backout state only
    """

    def __init__(self, parent_node):
        os.environ.setdefault("AIC_SC_VS_FINAL_CM_PROFILE", "gentle_commit")
        super().__init__(parent_node)
        self.anchor_waypoint = _int_env(
            "AIC_SC_ANCHOR_WAYPOINT",
            _int_env("AIC_SC_VS_START_WAYPOINT", 310),
        )
        self.late_diagnostic_min_waypoint = _int_env(
            "AIC_SC_ANCHOR_LATE_DIAGNOSTIC_MIN_WAYPOINT", 320
        )
        self.allow_late_blind_commit = (
            os.environ.get("AIC_SC_ANCHOR_ALLOW_LATE_BLIND_COMMIT", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        self.late_backout_distance_m = max(
            0.0,
            _float_env("AIC_SC_ANCHOR_LATE_BACKOUT_DISTANCE_M", 0.0030),
        )
        self.late_backout_hold_s = max(
            0.0,
            _float_env("AIC_SC_ANCHOR_LATE_BACKOUT_HOLD_S", 0.35),
        )
        self.pre_commit_hold_s = max(
            0.0,
            _float_env("AIC_SC_ANCHOR_PRE_COMMIT_HOLD_S", 0.0),
        )
        self.get_logger().info(
            "RunSCAnchorGentleCommit configured: "
            f"anchor_waypoint={self.anchor_waypoint} "
            f"late_diagnostic_min_waypoint={self.late_diagnostic_min_waypoint} "
            f"allow_late_blind_commit={self.allow_late_blind_commit} "
            f"late_backout_distance_m={self.late_backout_distance_m:.4f} "
            f"pre_commit_hold_s={self.pre_commit_hold_s:.2f} "
            f"final_cm_profile={self.final_centimeter_profile} "
            f"max_insert_steps={self.max_insert_steps} "
            f"insert_step_m={self.insert_step_m:.5f}"
        )

    def _is_late_diagnostic_anchor(self, anchor_index: int) -> bool:
        return int(anchor_index) >= int(self.late_diagnostic_min_waypoint)

    def _late_anchor_reseat_backout(
        self,
        *,
        anchor_index: int,
        anchor_pose,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> str:
        observation = get_observation()
        force_n = 0.0
        pose = self._copy_pose(anchor_pose)
        if observation is not None:
            pose = self._pose_from_observation(observation)
            force_n = self._force_from_observation(observation)

        insert_axis = self._template_insert_axis(anchor_index)
        backout_delta = -insert_axis * self.late_backout_distance_m
        target_pose = self._offset_pose_xyz(pose, backout_delta)
        self.set_pose_target(
            move_robot=move_robot,
            pose=target_pose,
            stiffness=self.stiffness,
            damping=self.damping,
        )
        send_feedback(
            "SC late-anchor diagnostic reseat/backout "
            f"anchor={anchor_index} backout_m={self.late_backout_distance_m:.4f} "
            f"force={force_n:.2f} events={len(self.insertion_events)}"
        )
        if self.late_backout_hold_s > 0.0:
            self.sleep_for(self.late_backout_hold_s)
        return (
            f"late_anchor_reseat_backout:"
            f"anchor={anchor_index}:force={force_n:.2f}:events={len(self.insertion_events)}"
        )

    def _replay_to_anchor(
        self,
        *,
        anchor_index: int,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> tuple[object, str]:
        hard_force_steps = 0
        last_pose = self.waypoints[0].pose
        for step, waypoint in enumerate(self.waypoints[: anchor_index + 1]):
            loop_start = time.time()
            observation = get_observation()
            if observation is not None:
                force_n = self._force_from_observation(observation)
            else:
                force_n = max(self.last_force_magnitude_n, self.max_force_magnitude_n)

            hard_force_steps = hard_force_steps + 1 if force_n >= self.hard_force_n else 0
            if hard_force_steps >= self.hard_force_consecutive_steps:
                reason = f"anchor_replay_hard_force:{force_n:.2f}"
                self.get_logger().error(
                    f"SC anchor replay abort step={step} force_n={force_n:.2f}"
                )
                return last_pose, reason

            self.set_pose_target(
                move_robot=move_robot,
                pose=waypoint.pose,
                stiffness=self.stiffness,
                damping=self.damping,
            )
            last_pose = waypoint.pose
            if step % 20 == 0:
                self.get_logger().info(
                    "SC anchor replay "
                    f"step={step}/{anchor_index} force_n={force_n:.2f} "
                    f"events={len(self.insertion_events)}"
                )
            send_feedback(
                f"SC anchor replay step={step} force={force_n:.2f} "
                f"events={len(self.insertion_events)}"
            )
            if self.stop_on_insertion_event and self.insertion_events:
                return last_pose, "anchor_replay_insertion_event"

            elapsed = time.time() - loop_start
            self.sleep_for(max(0.0, self.period_s - elapsed))

        observation = get_observation()
        if observation is not None:
            last_pose = self._pose_from_observation(observation)
            self._force_from_observation(observation)
        return last_pose, "anchor_reached"

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"RunSCAnchorGentleCommit.insert_cable() task: {self._task_summary(task)}"
        )
        self._select_replay_template_for_task(task)
        self.cv_pipeline.reset()
        start_time = time.time()

        anchor_index = int(np.clip(self.anchor_waypoint, 0, len(self.waypoints) - 1))
        anchor_pose, anchor_reason = self._replay_to_anchor(
            anchor_index=anchor_index,
            get_observation=get_observation,
            move_robot=move_robot,
            send_feedback=send_feedback,
        )
        anchor_metrics = self._anchor_handoff_metrics(
            task=task,
            get_observation=get_observation,
            start_index=anchor_index,
            label="anchor_gentle_commit_handoff",
        )

        insert_reason = "skipped"
        if anchor_reason in {"anchor_reached", "anchor_replay_insertion_event"}:
            should_run_commit = True
            if (
                self._is_late_diagnostic_anchor(anchor_index)
                and not self.allow_late_blind_commit
            ):
                should_run_commit = False
                insert_reason = self._late_anchor_reseat_backout(
                    anchor_index=anchor_index,
                    anchor_pose=anchor_pose,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                )

            if (
                should_run_commit
                and self.pre_commit_hold_s > 0.0
                and not self.insertion_events
            ):
                self._hold_pose(
                    move_robot=move_robot,
                    pose=anchor_pose,
                    hold_s=self.pre_commit_hold_s,
                    send_feedback=send_feedback,
                    reason="SC anchor pre-commit hold",
                )
                observation = get_observation()
                if observation is not None:
                    anchor_pose = self._pose_from_observation(observation)
                    force_n = self._force_from_observation(observation)
                    self.get_logger().info(
                        "SC anchor pre-commit hold complete: "
                        f"hold_s={self.pre_commit_hold_s:.2f} "
                        f"force_n={force_n:.2f} events={self.insertion_events}"
                    )

            if should_run_commit:
                insert_axis = self._template_insert_axis(anchor_index)
                insert_reason = self._feather_insert(
                    task=task,
                    get_observation=get_observation,
                    move_robot=move_robot,
                    send_feedback=send_feedback,
                    start_pose=anchor_pose,
                    insert_axis=insert_axis,
                )
        else:
            self.get_logger().warning(
                f"SC anchor gentle commit skipped after anchor_reason={anchor_reason}"
            )

        hold_pose = anchor_pose
        observation = get_observation()
        if observation is not None:
            hold_pose = self._pose_from_observation(observation)
            self._force_from_observation(observation)

        if self.final_hold_s > 0.0:
            self._hold_pose(
                move_robot=move_robot,
                pose=hold_pose,
                hold_s=self.final_hold_s,
                send_feedback=send_feedback,
                reason="SC anchor gentle commit final hold",
            )

        self.get_logger().info(
            "RunSCAnchorGentleCommit exiting: "
            f"template={self.template_name} anchor_index={anchor_index} "
            f"anchor_reason={anchor_reason} insert_reason={insert_reason} "
            f"anchor_metrics={anchor_metrics} "
            f"events={self.insertion_events} "
            f"max_force_n={self.max_force_magnitude_n:.2f} "
            f"duration_s={time.time() - start_time:.2f}"
        )
        return True
