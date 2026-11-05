"""RunSCVisualServoReplay, then the scripted wiggle-descent finish (v5).

The replay drives the recorded SC template unchanged. On scene
sc_scripted_full_001_026 it ends with the plug in the port about 1 cm short
of the seat. If no insertion event has arrived by then, this policy runs the
same finish as RunACTStagedScriptedInsert v5: compliant across base_link z
and firm along it, a slow descent with a small circular wiggle, and stall
retries that back off and re-descend at a 1 mm XY offset in a new direction.
No port pose is read. The finish loop is a copy of the v5 loop, adapted to the
replay base class (its force helper includes the resting cable load, so the
finish tares the raw wrench itself).
"""

from __future__ import annotations

import collections
import math
import os
import time

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Point, Pose, Vector3, Wrench, WrenchStamped
from rclpy.time import Time

from .ReplaySCVisualServo import RunSCVisualServoReplay

_RETRY_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _xyz(vec) -> list[float]:
    return [round(float(vec.x), 5), round(float(vec.y), 5), round(float(vec.z), 5)]


class RunSCVisualServoReplayScriptedInsert(RunSCVisualServoReplay):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._latest_wrench_msg: WrenchStamped | None = None
        self.insert_depth_m = _env_float("AIC_SCRIPTED_INSERT_DEPTH_M", 0.025)
        self.insert_speed_mps = _env_float("AIC_SCRIPTED_INSERT_SPEED_MPS", 0.004)
        self.insert_time_s = _env_float("AIC_SCRIPTED_INSERT_TIME_S", 50.0)
        self.stall_window_s = _env_float("AIC_SCRIPTED_INSERT_STALL_WINDOW_S", 2.0)
        self.stall_progress_m = _env_float(
            "AIC_SCRIPTED_INSERT_STALL_PROGRESS_M", 0.0005)
        self.min_lead_m = _env_float("AIC_SCRIPTED_INSERT_MIN_LEAD_M", 0.005)
        self.retreat_m = _env_float("AIC_SCRIPTED_INSERT_RETREAT_M", 0.010)
        self.retreat_hold_s = _env_float("AIC_SCRIPTED_INSERT_RETREAT_HOLD_S", 1.0)
        self.max_retries = int(_env_float("AIC_SCRIPTED_INSERT_RETRIES", 8))
        self.retry_offset_m = _env_float("AIC_SCRIPTED_INSERT_RETRY_OFFSET_M", 0.001)
        self.stop_force_n = _env_float("AIC_SCRIPTED_INSERT_STOP_FORCE_N", 15.0)
        self.wiggle_start_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_M", 0.0005)
        self.wiggle_max_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_MAX_M", 0.0025)
        self.wiggle_hz = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_HZ", 1.0)
        self.insert_period_s = _env_float("AIC_SCRIPTED_INSERT_PERIOD_S", 0.05)
        self.max_obs_age_s = _env_float("AIC_SCRIPTED_INSERT_MAX_OBS_AGE_S", 0.5)
        self.insert_stiffness_diag = [
            float(v) for v in os.environ.get(
                "AIC_SCRIPTED_INSERT_STIFFNESS",
                "150,150,500,85,85,85").split(",")]
        self.insert_damping_diag = [75.0] * 6

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        super()._wrench_cb(msg)
        self._latest_wrench_msg = msg

    def _tared_force_n(self, observation) -> float | None:
        if self._latest_wrench_msg is None:
            return None
        raw = self._latest_wrench_msg.wrench.force
        tare = observation.controller_state.fts_tare_offset.wrench.force
        return float(np.linalg.norm(
            [raw.x - tare.x, raw.y - tare.y, raw.z - tare.z]))

    def _obs_age_s(self, observation) -> float:
        stamp = Time.from_msg(observation.controller_state.header.stamp)
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _command(self, move_robot, pose: Pose, stiffness, damping) -> bool:
        msg = MotionUpdate()
        msg.pose = pose
        msg.header.frame_id = "base_link"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.target_stiffness = np.diag(stiffness).flatten()
        msg.target_damping = np.diag(damping).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0))
        msg.wrench_feedback_gains_at_tip = [0.0] * 6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
        return bool(move_robot(motion_update=msg))

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.insertion_events.clear()
        super().insert_cable(task, get_observation, move_robot, send_feedback)
        if self.insertion_events:
            return True
        return self._scripted_insert(get_observation, move_robot)

    def _scripted_insert(self, get_observation, move_robot) -> bool:
        observation = get_observation()
        start = observation.controller_state.tcp_pose
        start_xyz = np.array([start.position.x, start.position.y, start.position.z])
        self.get_logger().info(
            "Scripted insert hand-off after replay at "
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
        reason = "time_limit"
        try:
            while (now := time.time()) - t0 < self.insert_time_s:
                if self.insertion_events:
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
                force_n = self._tared_force_n(observation)
                if force_n is not None and force_n >= self.stop_force_n:
                    reason = "force_stop"
                    break
                if retreat_until is not None:
                    if now < retreat_until:
                        target_xyz = (anchor[0], anchor[1], retreat_z)
                    else:
                        # Descend again at the offset XY from the settled height.
                        anchor = np.array([anchor[0], anchor[1], tcp.z])
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
                        dx, dy = _RETRY_DIRECTIONS[(retries - 1) % 4]
                        radius = self.retry_offset_m * (1 + (retries - 1) // 4)
                        anchor = np.array([start_xyz[0] + radius * dx,
                                           start_xyz[1] + radius * dy,
                                           tcp.z])
                        retreat_z = tcp.z + self.retreat_m
                        retreat_until = now + self.retreat_hold_s
                        self.get_logger().info(
                            f"SI_RETRY {retries}/{self.max_retries} at "
                            f"tcp={_xyz(tcp)} target_z={target_z:.4f} "
                            f"force={force_n} offset_m={radius:.4f} dir={(dx, dy)}")
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
                accepted = self._command(
                    move_robot, target, self.insert_stiffness_diag,
                    self.insert_damping_diag)
                self.get_logger().info(
                    f"SI_TICK t={now - t0:.3f} req={_xyz(target.position)} "
                    f"acc={accepted} tcp={_xyz(tcp)} force={force_n}")
                time.sleep(max(0.0, self.insert_period_s - (time.time() - now)))
        finally:
            hold = get_observation()
            if hold is not None:
                self._command(move_robot, hold.controller_state.tcp_pose,
                              self.stiffness, self.damping)
        self.get_logger().info(
            f"Scripted insert done. reason={reason} "
            f"descended={(start_xyz[2] - min_z) * 1000:.2f}mm "
            f"retries={retries} final_wiggle={wiggle_m * 1000:.2f}mm "
            f"events={self.insertion_events}")
        return True
