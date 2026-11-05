"""RunACTStaged approach/align, then a scripted, ground-truth-free insertion.

The learned stages run unchanged, including the wrapper's guarded insert,
which pulls the plug onto the port axis but stalls ~13 mm above the seat.
When the wrapper would finish there, this policy takes over and descends
straight down in base_link from the stall pose at a slow, fixed rate. A
small circular wiggle helps the plug past the port lip; its radius grows only
while depth progress stalls. No port pose is read.

v2 (2026-10-27): the learned-stage stiffness is restored at every episode
start and on exit (the engine reuses this object across trials); hand-off
happens only on the benign wrapper reasons; every command result, the
controller's adopted reference, the raw and tared wrench and the observation
age are logged per tick; rejected commands and stale observations end the
finish; every exit sends a hold at the current TCP pose. The frozen v1 is in
aic-quantarune/policy_snapshots/RunACTStagedScriptedInsert_v1_20261027.py.

v3 (2026-10-27): the finish is compliant across the insertion axis (base_link
z for the flat SC board) and firm along it. A rolling stall test (little
descent over the last window while the target still leads) triggers a retry:
back off, settle, then descend again from where the compliant plug settled.
v2 logs showed the outcome was set at hand-off: plugs already ~1 mm past the
lip slid home; plugs on the lip jammed at ~12 N with every command accepted.
"""

import collections
import math
import os
import time

import numpy as np
from geometry_msgs.msg import Point, Pose
from rclpy.time import Time

from .RunACTStaged import RunACTStaged

_HANDOFF_REASONS = ("guarded_insert_complete", "no_progress")


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _xyz(vec) -> list[float]:
    return [round(float(vec.x), 5), round(float(vec.y), 5), round(float(vec.z), 5)]


class _HandOff(Exception):
    """Raised inside the parent loop when the guarded insert gives up."""


class RunACTStagedScriptedInsert(RunACTStaged):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._learned_stiffness_diag = list(self.control_stiffness_diag)
        self.insert_depth_m = _env_float("AIC_SCRIPTED_INSERT_DEPTH_M", 0.025)
        self.insert_speed_mps = _env_float("AIC_SCRIPTED_INSERT_SPEED_MPS", 0.004)
        self.insert_time_s = _env_float("AIC_SCRIPTED_INSERT_TIME_S", 30.0)
        self.stall_window_s = _env_float("AIC_SCRIPTED_INSERT_STALL_WINDOW_S", 2.0)
        self.stall_progress_m = _env_float(
            "AIC_SCRIPTED_INSERT_STALL_PROGRESS_M", 0.0005)
        self.min_lead_m = _env_float("AIC_SCRIPTED_INSERT_MIN_LEAD_M", 0.005)
        self.retreat_m = _env_float("AIC_SCRIPTED_INSERT_RETREAT_M", 0.010)
        self.retreat_hold_s = _env_float("AIC_SCRIPTED_INSERT_RETREAT_HOLD_S", 1.0)
        self.max_retries = int(_env_float("AIC_SCRIPTED_INSERT_RETRIES", 3))
        self.stop_force_n = _env_float("AIC_SCRIPTED_INSERT_STOP_FORCE_N", 15.0)
        self.wiggle_start_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_M", 0.0005)
        self.wiggle_max_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_MAX_M", 0.0025)
        self.wiggle_hz = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_HZ", 1.0)
        self.period_s = _env_float("AIC_SCRIPTED_INSERT_PERIOD_S", 0.05)
        self.max_obs_age_s = _env_float("AIC_SCRIPTED_INSERT_MAX_OBS_AGE_S", 0.5)
        self.max_rejects = int(_env_float("AIC_SCRIPTED_INSERT_MAX_REJECTS", 3))
        # The learned stages run at 85 N/m. The finish is soft across the
        # insertion axis, so the chamfer can steer the plug, and firm along it.
        self.insert_stiffness_diag = [
            float(v) for v in os.environ.get(
                "AIC_SCRIPTED_INSERT_STIFFNESS",
                "150,150,500,85,85,85").split(",")]

    def _maybe_finish_without_event(self, reason: str) -> bool:
        result = super()._maybe_finish_without_event(reason)
        if (reason in _HANDOFF_REASONS
                and self._phase in ("GUARDED_INSERT", "RETRY_OR_FINISH")):
            raise _HandOff()
        return result

    def insert_cable(self, task, get_observation, move_robot, send_feedback,
                     **kwargs):
        self.control_stiffness_diag = list(self._learned_stiffness_diag)
        try:
            return super().insert_cable(
                task, get_observation, move_robot, send_feedback, **kwargs)
        except _HandOff:
            return self._scripted_insert(get_observation, move_robot)

    def _obs_age_s(self, observation) -> float:
        stamp = Time.from_msg(observation.controller_state.header.stamp)
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _log_tick(self, t, target, accepted, observation, force_n):
        state = observation.controller_state
        raw = self._latest_wrench_msg
        self.get_logger().info(
            f"SI_TICK t={t:.3f} req={_xyz(target.position)} acc={accepted} "
            f"tcp={_xyz(state.tcp_pose.position)} "
            f"ref={_xyz(state.reference_tcp_pose.position)} "
            f"raw_f={_xyz(raw.wrench.force) if raw is not None else None} "
            f"tare_f={_xyz(state.fts_tare_offset.wrench.force)} "
            f"comp_f={force_n if force_n is None else round(force_n, 3)} "
            f"age={self._obs_age_s(observation):.3f} "
            f"stiff={self.control_stiffness_diag[:3]}")

    def _hold_current_pose(self, get_observation, move_robot) -> None:
        observation = get_observation()
        if observation is None:
            self.get_logger().warning("Scripted insert exit: no observation to hold.")
            return
        hold = Pose(
            position=observation.controller_state.tcp_pose.position,
            orientation=observation.controller_state.tcp_pose.orientation)
        accepted = bool(
            move_robot(motion_update=self.create_pose_motion_update(hold)))
        self.get_logger().info(
            f"Scripted insert exit hold at tcp={_xyz(hold.position)} "
            f"acc={accepted} stiff={self.control_stiffness_diag[:3]}")

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
        floor_z = start_xyz[2] - self.insert_depth_m
        anchor = start_xyz.copy()
        wiggle_m = self.wiggle_start_m
        best_z = min_z = start_xyz[2]
        last_gain_t = t0 = attempt_t0 = stale_since = time.time()
        history = collections.deque()
        retries, retreat_until, retreat_z = 0, None, None
        rejects = 0
        reason = "time_limit"
        self.control_stiffness_diag = list(self.insert_stiffness_diag)
        try:
            while (now := time.time()) - t0 < self.insert_time_s:
                if self._insertion_event_seen:
                    reason = "insertion_event"
                    break
                observation = get_observation()
                if observation is None:
                    continue
                if self._obs_age_s(observation) <= self.max_obs_age_s:
                    stale_since = now
                elif now - stale_since > 1.0:
                    reason = "stale_observation"
                    break
                tcp = observation.controller_state.tcp_pose.position
                min_z = min(min_z, tcp.z)
                force_n = self._compensated_force_magnitude(observation)
                if force_n is not None and force_n >= self.stop_force_n:
                    reason = "force_stop"
                    break
                if retreat_until is not None:
                    if now < retreat_until:
                        target_xyz = (anchor[0], anchor[1], retreat_z)
                    else:
                        # Descend again from where the compliant plug settled.
                        anchor = np.array([tcp.x, tcp.y, tcp.z])
                        attempt_t0, last_gain_t = now, now
                        best_z, wiggle_m = tcp.z, self.wiggle_start_m
                        history.clear()
                        retreat_until = None
                if retreat_until is None:
                    history.append((now, tcp.z))
                    while history and now - history[0][0] > self.stall_window_s:
                        history.popleft()
                    target_z = max(
                        floor_z,
                        anchor[2] - self.insert_speed_mps * (now - attempt_t0))
                    stalled = (
                        now - attempt_t0 > self.stall_window_s
                        and history[0][1] - tcp.z < self.stall_progress_m
                        and tcp.z - target_z > self.min_lead_m)
                    if stalled and retries < self.max_retries:
                        retries += 1
                        anchor = np.array([tcp.x, tcp.y, tcp.z])
                        retreat_z = tcp.z + self.retreat_m
                        retreat_until = now + self.retreat_hold_s
                        self.get_logger().info(
                            f"SI_RETRY {retries}/{self.max_retries} at "
                            f"tcp={_xyz(tcp)} target_z={target_z:.4f} "
                            f"force={force_n}")
                        target_xyz = (anchor[0], anchor[1], retreat_z)
                    else:
                        if tcp.z < best_z - 0.0005:
                            best_z, last_gain_t = tcp.z, now
                        elif now - last_gain_t > 1.0:
                            wiggle_m = min(wiggle_m + 0.00025, self.wiggle_max_m)
                            last_gain_t = now
                        phase = 2.0 * math.pi * self.wiggle_hz * (now - attempt_t0)
                        target_xyz = (anchor[0] + wiggle_m * math.cos(phase),
                                      anchor[1] + wiggle_m * math.sin(phase),
                                      target_z)
                target = Pose(
                    position=Point(x=float(target_xyz[0]), y=float(target_xyz[1]),
                                   z=float(target_xyz[2])),
                    orientation=start.orientation)
                accepted = bool(
                    move_robot(motion_update=self.create_pose_motion_update(target)))
                self._log_tick(now - t0, target, accepted, observation, force_n)
                rejects = 0 if accepted else rejects + 1
                if rejects >= self.max_rejects:
                    reason = "command_rejected"
                    break
                time.sleep(max(0.0, self.period_s - (time.time() - now)))
        finally:
            self.control_stiffness_diag = list(self._learned_stiffness_diag)
            self._hold_current_pose(get_observation, move_robot)
        self.get_logger().info(
            f"Scripted insert done. reason={reason} "
            f"descended={(start_xyz[2] - min_z) * 1000:.2f}mm "
            f"retries={retries} final_wiggle={wiggle_m * 1000:.2f}mm")
        self._finalize_canonical_episode(
            wrapper_returned_success=True, termination_reason=reason)
        return True
