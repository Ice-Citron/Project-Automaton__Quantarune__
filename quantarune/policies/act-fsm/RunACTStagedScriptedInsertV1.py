"""Frozen v1 of RunACTStagedScriptedInsert (policy_snapshots/RunACTStagedScriptedInsert_v1_20261027.py), loadable by name."""

"""RunACTStaged approach/align, then a scripted, ground-truth-free insertion.

The learned stages run unchanged, including the wrapper's guarded insert,
which pulls the plug onto the port axis but stalls ~13 mm above the seat.
When the wrapper would finish there, this policy takes over and descends
straight down in base_link from the stall pose at a slow, fixed rate. A
small circular wiggle helps the plug past the port lip; its radius grows only
while depth progress stalls. No port pose is read.
"""

import math
import os
import time

import numpy as np
from geometry_msgs.msg import Point, Pose

from .RunACTStaged import RunACTStaged


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


class _HandOff(Exception):
    """Raised inside the parent loop when the guarded insert gives up."""


class RunACTStagedScriptedInsertV1(RunACTStaged):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.insert_depth_m = _env_float("AIC_SCRIPTED_INSERT_DEPTH_M", 0.025)
        self.insert_speed_mps = _env_float("AIC_SCRIPTED_INSERT_SPEED_MPS", 0.004)
        self.insert_time_s = _env_float("AIC_SCRIPTED_INSERT_TIME_S", 15.0)
        self.stop_force_n = _env_float("AIC_SCRIPTED_INSERT_STOP_FORCE_N", 15.0)
        self.wiggle_start_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_M", 0.0005)
        self.wiggle_max_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_MAX_M", 0.0025)
        self.wiggle_hz = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_HZ", 1.0)
        self.period_s = _env_float("AIC_SCRIPTED_INSERT_PERIOD_S", 0.05)
        # The learned stages run at 85 N/m; that is too soft to move a plug
        # held on the port lip, so only the scripted finish runs stiffer.
        self.insert_stiffness_diag = [
            float(v) for v in os.environ.get(
                "AIC_SCRIPTED_INSERT_STIFFNESS",
                "400,400,400,85,85,85").split(",")]

    def _maybe_finish_without_event(self, reason: str) -> bool:
        result = super()._maybe_finish_without_event(reason)
        if self._phase in ("GUARDED_INSERT", "RETRY_OR_FINISH"):
            raise _HandOff()
        return result

    def insert_cable(self, task, get_observation, move_robot, send_feedback,
                     **kwargs):
        try:
            return super().insert_cable(
                task, get_observation, move_robot, send_feedback, **kwargs)
        except _HandOff:
            return self._scripted_insert(get_observation, move_robot)

    def _scripted_insert(self, get_observation, move_robot) -> bool:
        observation = get_observation()
        start = observation.controller_state.tcp_pose
        start_xyz = np.array(
            [start.position.x, start.position.y, start.position.z])
        self.get_logger().info(
            "Scripted insert hand-off at "
            f"xyz={np.round(start_xyz, 4).tolist()} "
            f"depth={self.insert_depth_m} speed={self.insert_speed_mps} "
            f"stiffness={self.insert_stiffness_diag}")
        self.control_stiffness_diag = self.insert_stiffness_diag
        wiggle_m = self.wiggle_start_m
        best_z = start_xyz[2]
        last_gain_t = t0 = time.time()
        reason = "time_limit"
        while (now := time.time()) - t0 < self.insert_time_s:
            if self._insertion_event_seen:
                reason = "insertion_event"
                break
            observation = get_observation()
            if observation is None:
                continue
            tcp_z = observation.controller_state.tcp_pose.position.z
            if tcp_z < best_z - 0.0005:
                best_z, last_gain_t = tcp_z, now
            elif now - last_gain_t > 1.0:
                wiggle_m = min(wiggle_m + 0.00025, self.wiggle_max_m)
                last_gain_t = now
            force_n = self._compensated_force_magnitude(observation)
            if force_n is not None and force_n >= self.stop_force_n:
                reason = "force_stop"
                break
            depth = min(self.insert_speed_mps * (now - t0), self.insert_depth_m)
            phase = 2.0 * math.pi * self.wiggle_hz * (now - t0)
            target = Pose(
                position=Point(
                    x=float(start_xyz[0] + wiggle_m * math.cos(phase)),
                    y=float(start_xyz[1] + wiggle_m * math.sin(phase)),
                    z=float(start_xyz[2] - depth)),
                orientation=start.orientation)
            move_robot(motion_update=self.create_pose_motion_update(target))
            time.sleep(max(0.0, self.period_s - (time.time() - now)))
        self.get_logger().info(
            f"Scripted insert done. reason={reason} "
            f"descended={(start_xyz[2] - best_z) * 1000:.1f}mm "
            f"final_wiggle={wiggle_m * 1000:.2f}mm")
        self._finalize_canonical_episode(
            wrapper_returned_success=True, termination_reason=reason)
        return True
