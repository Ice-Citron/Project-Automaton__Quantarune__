"""MegaPose-guided approach with the scripted insert, ground truth off.

Look-move-look: a centre-camera frame goes to the public MegaPose Space
with a box from the classical port finder. The returned mesh pose becomes
the port pose through a rigid calibration measured once on a ground-truth
episode of SFP scene 137 (2026-11-04). The gripper centres the plug tip over
the estimated port and descends in bounded steps until the camera is close
enough for the estimate to be trusted, then hands over to the v4 scripted
insert. No port transform is read; robot transforms only.
"""
import collections
import json
import math
import os
import subprocess
import tempfile
import time

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion, WrenchStamped
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from aic_task_interfaces.action import InsertCable
from PIL import Image

from .CheatCode import CheatCode
from .vision_target_features import ros_image_to_rgb
from .template_port_detector import TemplatePortDetector

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_SCRATCH = os.path.dirname(_PKG)
_VENV_PY = os.path.join(_SCRATCH, "megapose_venv", "bin", "python")
_SPACE_SCRIPT = os.path.join(_SCRATCH, "megapose_space", "space_pose.py")
_RETRY_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tf_to_mat(transform):
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(transform.rotation)
    t = transform.translation
    m[:3, 3] = [t.x, t.y, t.z]
    return m


def _pose_to_mat(pose):
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(pose.orientation)
    m[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return m


def _yaw_quat_xyzw(yaw):
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def _quat_mul_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz])


def _xyz(p):
    return [round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)]


class RunMegaPoseScriptedInsert(CheatCode):

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.mesh_path = os.environ.get("AIC_MEGAPOSE_MESH", os.path.join(_PKG, "sfp_mount.obj"))
        calib = json.load(open(os.environ.get(
            "AIC_MEGAPOSE_CALIB", os.path.join(_PKG, "calib_sfp_mount_to_port.json"))))
        self.T_mesh_port = np.array(calib["T_mesh_port"])
        # Plug tip in the TCP frame and plug yaw relative to the TCP x axis,
        # measured on the ground-truth episode of 2026-11-04 (scene 137).
        self.tip_in_tcp = np.array([0.0, -0.0207, 0.0541])
        self.plug_yaw_offset = math.radians(3.0)
        tip_json = os.path.join(_PKG, "tip_in_tcp.json")
        self.T_tcp_tip = np.array(json.load(open(tip_json))["T_tcp_tip"]) if os.path.exists(tip_json) else None
        self.align_plug_axis = os.environ.get("AIC_MEGAPOSE_ALIGN_PLUG_AXIS", "1") == "1"
        self.port_depth_m = 0.0458          # entrance sits this far out along -z of the port link
        self.trust_depth_m = _env_float("AIC_MEGAPOSE_TRUST_DEPTH_M", 0.33)
        self.agree_m = _env_float("AIC_MEGAPOSE_AGREE_M", 0.015)
        self.max_looks = int(_env_float("AIC_MEGAPOSE_MAX_LOOKS", 5))
        self.step_cap_m = _env_float("AIC_MEGAPOSE_STEP_CAP_M", 0.035)
        self.keep_gap_m = _env_float("AIC_MEGAPOSE_KEEP_GAP_M", 0.03)
        self.handoff_gap_m = _env_float("AIC_MEGAPOSE_HANDOFF_GAP_M", 0.025)
        self.move_settle_s = _env_float("AIC_MEGAPOSE_MOVE_SETTLE_S", 3.0)
        self.box_pad_frac = _env_float("AIC_MEGAPOSE_BOX_PAD_FRAC", 0.0)
        self.space_timeout_s = _env_float("AIC_MEGAPOSE_SPACE_TIMEOUT_S", 150.0)
        self.log_dir = os.environ.get("AIC_MEGAPOSE_LOG_DIR") or tempfile.mkdtemp(prefix="megapose_policy_")
        os.makedirs(self.log_dir, exist_ok=True)
        # scripted insert (v4 defaults, deeper floor for the SFP port)
        self.insert_depth_m = _env_float("AIC_SCRIPTED_INSERT_DEPTH_M", 0.060)
        self.insert_speed_mps = _env_float("AIC_SCRIPTED_INSERT_SPEED_MPS", 0.004)
        self.insert_time_s = _env_float("AIC_SCRIPTED_INSERT_TIME_S", 30.0)
        self.stall_window_s = _env_float("AIC_SCRIPTED_INSERT_STALL_WINDOW_S", 2.0)
        self.stall_progress_m = _env_float("AIC_SCRIPTED_INSERT_STALL_PROGRESS_M", 0.0005)
        self.min_lead_m = _env_float("AIC_SCRIPTED_INSERT_MIN_LEAD_M", 0.005)
        self.retreat_m = _env_float("AIC_SCRIPTED_INSERT_RETREAT_M", 0.010)
        self.retreat_hold_s = _env_float("AIC_SCRIPTED_INSERT_RETREAT_HOLD_S", 1.0)
        self.max_retries = int(_env_float("AIC_SCRIPTED_INSERT_RETRIES", 4))
        self.retry_offset_m = _env_float("AIC_SCRIPTED_INSERT_RETRY_OFFSET_M", 0.001)
        self.stop_force_n = _env_float("AIC_SCRIPTED_INSERT_STOP_FORCE_N", 15.0)
        self.wiggle_start_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_M", 0.0005)
        self.wiggle_max_m = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_MAX_M", 0.0025)
        self.wiggle_hz = _env_float("AIC_SCRIPTED_INSERT_WIGGLE_HZ", 1.0)
        self.period_s = _env_float("AIC_SCRIPTED_INSERT_PERIOD_S", 0.05)
        self.insert_stiffness = [float(v) for v in os.environ.get(
            "AIC_SCRIPTED_INSERT_STIFFNESS", "150,150,500,85,85,85").split(",")]
        self.insert_damping = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
        self.insert_xy_gain = _env_float("AIC_SCRIPTED_INSERT_XY_GAIN", 0.15)
        self._latest_wrench = None
        self._insertion_event_seen = False
        self._last_event = ""
        self._parent_node.create_subscription(
            WrenchStamped, "/fts_broadcaster/wrench", self._on_wrench, 10)
        self._parent_node.create_subscription(
            String, "/scoring/insertion_event", self._on_event, 10)
        bank = os.environ.get("AIC_MEGAPOSE_TEMPLATES", os.path.join(_PKG, "templates_sfp137"))
        self._detector = TemplatePortDetector(bank)
        self.taught_port_yaw = math.radians(json.load(open(os.path.join(bank, "templates.json"))).get(
            "taught_port_yaw_deg", 0.0))
        self.template_yaw_search_deg = _env_float("AIC_MEGAPOSE_TEMPLATE_YAW_SEARCH_DEG", 20.0)
        self.yaw_source = os.environ.get("AIC_MEGAPOSE_YAW_SOURCE", "template")  # template | megapose | none
        self.min_box_score = _env_float("AIC_MEGAPOSE_MIN_BOX_SCORE", 0.55)
        self.depth_min_m = _env_float("AIC_MEGAPOSE_DEPTH_MIN_M", 0.18)
        self.depth_max_m = _env_float("AIC_MEGAPOSE_DEPTH_MAX_M", 0.60)
        self.lateral_cap_m = _env_float("AIC_MEGAPOSE_LATERAL_CAP_M", 0.05)
        self.max_gap_m = _env_float("AIC_MEGAPOSE_MAX_GAP_M", 0.20)
        self.blind_descend_m = _env_float("AIC_MEGAPOSE_BLIND_DESCEND_M", 0.015)
        self.megapose_depth_tol_frac = _env_float("AIC_MEGAPOSE_DEPTH_TOL_FRAC", 0.25)
        self.megapose_axis_min_cos = _env_float("AIC_MEGAPOSE_AXIS_MIN_COS", 0.9)
        self.megapose_xy_tol_m = _env_float("AIC_MEGAPOSE_XY_TOL_M", 0.03)
        self.megapose_yaw_max_rad = math.radians(_env_float("AIC_MEGAPOSE_YAW_MAX_DEG", 20.0))
        self.megapose_use = os.environ.get("AIC_MEGAPOSE_USE", "both")  # both | position | yaw | none
        self.track_rounds = int(_env_float("AIC_MEGAPOSE_TRACK_ROUNDS", 6))
        self.track_tol_m = _env_float("AIC_MEGAPOSE_TRACK_TOL_M", 0.0008)
        self.track_gain = _env_float("AIC_MEGAPOSE_TRACK_GAIN", 0.8)
        self.track_settle_s = _env_float("AIC_MEGAPOSE_TRACK_SETTLE_S", 1.0)
        self.track_force_stop_n = _env_float("AIC_MEGAPOSE_TRACK_FORCE_STOP_N", 5.0)
        self.servo_rounds = int(_env_float("AIC_MEGAPOSE_SERVO_ROUNDS", 4))
        self.servo_tol_px = _env_float("AIC_MEGAPOSE_SERVO_TOL_PX", 2.5)
        self.servo_max_px = _env_float("AIC_MEGAPOSE_SERVO_MAX_PX", 60.0)
        # fixed hand-off bias in base XY, calibrated on the ground-truth diagnostic run 014 of 2026-11-04
        self.handoff_bias_m = np.array([float(v) for v in os.environ.get(
            "AIC_MEGAPOSE_HANDOFF_BIAS_XY_MM", "0,0").split(",")]) / 1000.0
        self.get_logger().info(
            f"RunMegaPoseScriptedInsert mesh={self.mesh_path} log_dir={self.log_dir} use={self.megapose_use} "
            f"trust_depth={self.trust_depth_m} looks<={self.max_looks}")

    def _on_wrench(self, msg):
        self._latest_wrench = msg

    def _on_event(self, msg):
        self._insertion_event_seen = True
        self._last_event = msg.data

    # ---- perception -------------------------------------------------------

    def _camera(self, observation, k):
        """Frame, intrinsics and camera pose in base for one look. None if TF is missing."""
        rgb = ros_image_to_rgb(observation.center_image)
        png = os.path.join(self.log_dir, f"look{k}.png")
        Image.fromarray(np.ascontiguousarray(rgb)).save(png)
        K = np.array(observation.center_camera_info.k, dtype=float).reshape(3, 3)
        try:
            T_base_cam = _tf_to_mat(self._parent_node._tf_buffer.lookup_transform(
                "base_link", "center_camera/optical", Time()).transform)
        except TransformException as ex:
            self.get_logger().error(f"LOOK{k}: camera transform missing: {ex}")
            return None
        return rgb, png, K, T_base_cam

    def _template_estimate(self, rgb, K, T_base_cam, k, search_yaw=False):
        """Port position in base from the template box centre ray and template range."""
        angles = (0.0,)
        if search_yaw and self.template_yaw_search_deg > 0:
            n = int(self.template_yaw_search_deg // 4)
            angles = tuple(4.0 * i for i in range(-n, n + 1))
        det = self._detector.detect_with_range(rgb, min_score=self.min_box_score, angles_deg=angles)
        if det is None:
            self.get_logger().warning(f"LOOK{k}: template detector found no cage")
            return None
        b = det["box"]
        u, v = det["port_px"]
        ray = np.linalg.inv(K) @ np.array([u, v, 1.0])
        p_cam = ray / ray[2] * det["range_m"]
        p_base = T_base_cam[:3, :3] @ p_cam + T_base_cam[:3, 3]
        det.update({"port_cam": p_cam.tolist(), "port_base": p_base.tolist()})
        self.get_logger().info(
            f"LOOK{k}: template box={[round(x, 1) for x in b]} score={det['score']:.2f} "
            f"tpl={det['template']} scale={det['scale']:.2f} range={det['range_m']:.3f} "
            f"rot={det.get('rotation_deg', 0.0):.1f}deg port_base={np.round(p_base, 4).tolist()}")
        return det

    def _megapose_estimate(self, png, box, K, T_base_cam, k):
        """One MegaPose call through the Space. Returns (T_base_port, cam_depth) or None."""
        out = os.path.join(self.log_dir, f"look{k}_megapose.json")
        cmd = [_VENV_PY, "-I", _SPACE_SCRIPT, png, *[str(v) for v in box],
               str(K[0, 0]), str(K[1, 1]), str(K[0, 2]), str(K[1, 2]), self.mesh_path, "m", out]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.space_timeout_s)
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"LOOK{k}: Space call timed out after {self.space_timeout_s}s")
            return None
        if proc.returncode != 0 or not os.path.exists(out):
            self.get_logger().error(f"LOOK{k}: Space call failed rc={proc.returncode} "
                                    f"stderr={proc.stderr[-300:]!r}")
            return None
        res = json.load(open(out))
        T_cam_port = np.array(res["pose_T_cam_obj"]) @ self.T_mesh_port
        T_base_port = T_base_cam @ T_cam_port
        self.get_logger().info(
            f"LOOK{k}: megapose space_s={time.time() - t0:.1f} cam_depth={T_cam_port[2, 3]:.3f} "
            f"port_base={np.round(T_base_port[:3, 3], 4).tolist()} "
            f"port_z_axis={np.round(T_base_port[:3, 2], 3).tolist()}")
        res.update({"box": box, "T_base_cam": T_base_cam.tolist(), "T_base_port": T_base_port.tolist(),
                    "cam_depth_m": float(T_cam_port[2, 3])})
        json.dump(res, open(out, "w"), indent=1)
        return T_base_port, float(T_cam_port[2, 3])

    # ---- motion helpers ---------------------------------------------------

    def _tcp_pose(self, observation):
        return observation.controller_state.tcp_pose

    def _tcp_quat_for_port_yaw(self, port_yaw):
        """TCP orientation (xyzw) that makes the plug tip frame coincide with a
        port frame whose x axis has yaw port_yaw and whose z axis points down."""
        x = np.array([math.cos(port_yaw), math.sin(port_yaw), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        y = np.cross(z, x)
        R_port = np.stack([x, y, z], axis=1)
        R_tcp = R_port @ self.T_tcp_tip[:3, :3].T
        # rotation matrix -> quaternion xyzw
        m = R_tcp
        tr = np.trace(m)
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
        return q / np.linalg.norm(q)

    def _tcp_target_for_tip(self, tip_xyz, yaw_quat_xyzw):
        """TCP pose that puts the plug tip at tip_xyz with the given TCP orientation."""
        R = _quat_to_mat(Quaternion(x=yaw_quat_xyzw[0], y=yaw_quat_xyzw[1],
                                    z=yaw_quat_xyzw[2], w=yaw_quat_xyzw[3]))
        tcp = np.asarray(tip_xyz) - R @ self.tip_in_tcp
        return Pose(position=Point(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2])),
                    orientation=Quaternion(x=float(yaw_quat_xyzw[0]), y=float(yaw_quat_xyzw[1]),
                                           z=float(yaw_quat_xyzw[2]), w=float(yaw_quat_xyzw[3])))

    def _move_to(self, move_robot, get_observation, pose, steps=40, sleep_s=0.05, correct=True):
        """Interpolate the TCP to pose, then trim the commanded target by the
        measured tracking error so the compliant controller settles on the
        requested point (gravity and cable pull otherwise leave 3 to 7 mm)."""
        obs = get_observation()
        start = self._tcp_pose(obs)
        s = np.array([start.position.x, start.position.y, start.position.z])
        e = np.array([pose.position.x, pose.position.y, pose.position.z])
        for i in range(1, steps + 1):
            f = i / steps
            p = s + f * (e - s)
            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2])),
                          orientation=pose.orientation),
                stiffness=self._approach_stiffness, damping=self._approach_damping)
            self.sleep_for(sleep_s)
        self.sleep_for(self.move_settle_s)
        if not correct:
            return
        cmd = e.copy()
        for r in range(self.track_rounds):
            obs = get_observation()
            m = self._tcp_pose(obs).position
            err = e - np.array([m.x, m.y, m.z])
            if float(np.linalg.norm(err)) <= self.track_tol_m:
                break
            force_n = self._force_n(obs)
            if force_n is not None and force_n >= self.track_force_stop_n:
                self.get_logger().warning(f"TRACK: contact force {force_n:.1f} N, trim stopped")
                break
            cmd = cmd + self.track_gain * err
            self.set_pose_target(
                move_robot=move_robot,
                pose=Pose(position=Point(x=float(cmd[0]), y=float(cmd[1]), z=float(cmd[2])),
                          orientation=pose.orientation),
                stiffness=self._approach_stiffness, damping=self._approach_damping)
            self.sleep_for(self.track_settle_s)
        m = self._tcp_pose(get_observation()).position
        err = e - np.array([m.x, m.y, m.z])
        self.get_logger().info(
            f"TRACK: target={np.round(e, 4).tolist()} reached={[round(m.x, 4), round(m.y, 4), round(m.z, 4)]} "
            f"residual_mm={np.round(err * 1000, 1).tolist()} cmd_offset_mm={np.round((cmd - e) * 1000, 1).tolist()}")

    def _tip_xyz(self, observation):
        T = _pose_to_mat(self._tcp_pose(observation))
        return T[:3, 3] + T[:3, :3] @ self.tip_in_tcp

    # ---- main --------------------------------------------------------------

    def insert_cable(self, task, get_observation, move_robot, send_feedback, **kwargs):
        self.get_logger().info(f"RunMegaPoseScriptedInsert.insert_cable() task: {task}")
        self._task = task
        self._task_pub.publish(task)
        self._episode_pub.publish(String(data="start"))
        if self._tare_ft_client.wait_for_service(timeout_sec=5.0):
            future = self._tare_ft_client.call_async(Trigger.Request())
            self.sleep_for(1.0)
            if not (future.done() and future.result().success):
                self.get_logger().warn("F/T sensor tare may not have completed")
        else:
            self.get_logger().warn("F/T tare service not available, skipping")
        if not self._wait_for_tf("base_link", "center_camera/optical"):
            return self._finish(False, "camera transform unavailable")

        obs = get_observation()
        start_q = self._tcp_pose(obs).orientation
        q_tcp = np.array([start_q.x, start_q.y, start_q.z, start_q.w])
        tcp_yaw0 = math.atan2(2 * (start_q.w * start_q.z + start_q.x * start_q.y),
                              1 - 2 * (start_q.y ** 2 + start_q.z ** 2))
        down = np.array([0.0, 0.0, -1.0])

        # Stage A: coarse approach on the template ray and range, no MegaPose.
        entrance_tpl = None
        k = 0
        for k in range(1, self.max_looks + 1):
            obs = get_observation()
            cam = self._camera(obs, k)
            if cam is None:
                return self._finish(False, "camera transform unavailable")
            rgb, png, K, T_base_cam = cam
            tip = self._tip_xyz(obs)
            det = self._template_estimate(rgb, K, T_base_cam, k)
            if det is None:
                self._move_to(move_robot, get_observation, self._tcp_target_for_tip(
                    tip + np.array([0.0, 0.0, -self.blind_descend_m]), q_tcp))
                continue
            port = np.array(det["port_base"])
            entrance_tpl = port - self.port_depth_m * down
            gap = float(tip[2] - entrance_tpl[2])
            lateral = float(np.linalg.norm(entrance_tpl[:2] - tip[:2]))
            self.get_logger().info(
                f"LOOK{k}: entrance_tpl={np.round(entrance_tpl, 4).tolist()} tip={np.round(tip, 4).tolist()} "
                f"gap={gap:.4f} lateral={lateral:.4f}")
            if not (-0.02 <= gap <= self.max_gap_m and lateral <= 0.15):
                self.get_logger().warning(f"LOOK{k}: implausible template estimate; blind descent")
                self._move_to(move_robot, get_observation, self._tcp_target_for_tip(
                    tip + np.array([0.0, 0.0, -self.blind_descend_m]), q_tcp))
                continue
            if gap <= self.keep_gap_m + 0.005 and lateral <= 0.005:
                break
            descend = min(self.step_cap_m, max(0.0, gap - self.keep_gap_m))
            dxy = entrance_tpl[:2] - tip[:2]
            if lateral > self.lateral_cap_m:
                dxy = dxy * (self.lateral_cap_m / lateral)
            target_tip = np.array([tip[0] + dxy[0], tip[1] + dxy[1], tip[2] - descend])
            self.get_logger().info(f"MOVE{k}: tip -> {np.round(target_tip, 4).tolist()} (descend {descend:.3f})")
            self._move_to(move_robot, get_observation, self._tcp_target_for_tip(target_tip, q_tcp))
        if entrance_tpl is None:
            return self._finish(False, "no_template_detection")

        # Stage B: one MegaPose look from the close pose with the tight template box.
        obs = get_observation()
        k += 1
        cam = self._camera(obs, k)
        if cam is None:
            return self._finish(False, "camera transform unavailable")
        rgb, png, K, T_base_cam = cam
        tip = self._tip_xyz(obs)
        det = self._template_estimate(rgb, K, T_base_cam, k)
        source = "template"
        entrance = entrance_tpl
        dyaw = 0.0
        if det is not None:
            entrance = np.array(det["port_base"]) - self.port_depth_m * down
            if self.yaw_source == "template":
                # image rotation of the match (template and scale fixed) approximates
                # the port yaw change since teaching
                n = int(self.template_yaw_search_deg // 2)
                rot_deg, rot_score = self._detector.detect_rotation(
                    rgb, det["template"], det["scale"], tuple(2.0 * i for i in range(-n, n + 1)))
                self.get_logger().info(f"LOOK{k}: rotation search {rot_deg:.1f}deg score={rot_score:.2f}")
                port_yaw_tpl = self.taught_port_yaw + math.radians(rot_deg)
                dyaw = (port_yaw_tpl - self.plug_yaw_offset - tcp_yaw0 + math.pi) % (2 * math.pi) - math.pi
                self.get_logger().info(f"LOOK{k}: template yaw {math.degrees(port_yaw_tpl):.1f}deg -> dyaw {math.degrees(dyaw):.1f}deg")
            mp = self._megapose_estimate(png, det["box"], K, T_base_cam, k)
            if mp is not None:
                T_base_port, cam_depth = mp
                z_axis = T_base_port[:3, 2]
                depth_ok = abs(cam_depth - det["range_m"]) <= self.megapose_depth_tol_frac * det["range_m"]
                axis_ok = float(np.dot(z_axis, down)) >= self.megapose_axis_min_cos
                pos_mp = T_base_port[:3, 3]
                near_ok = float(np.linalg.norm(pos_mp[:2] - np.array(det["port_base"])[:2])) <= self.megapose_xy_tol_m
                x_axis = T_base_port[:3, 0]
                port_yaw = math.atan2(x_axis[1], x_axis[0])
                dyaw_mp = (port_yaw - self.plug_yaw_offset - tcp_yaw0 + math.pi) % (2 * math.pi) - math.pi
                self.get_logger().info(
                    f"LOOK{k}: megapose checks depth_ok={depth_ok} axis_ok={axis_ok} near_ok={near_ok} "
                    f"dyaw={math.degrees(dyaw_mp):.1f}deg")
                if depth_ok and axis_ok and near_ok:
                    if self.megapose_use in ("both", "position"):
                        entrance = pos_mp - self.port_depth_m * z_axis
                        source = "megapose_position"
                    if (self.megapose_use in ("both", "yaw") and self.yaw_source == "megapose"
                            and abs(dyaw_mp) <= self.megapose_yaw_max_rad):
                        dyaw = dyaw_mp
                        source = "megapose_both" if source.startswith("megapose") else "megapose_yaw"
        q_final = _quat_mul_xyzw(_yaw_quat_xyzw(dyaw), q_tcp)
        if self.align_plug_axis and self.T_tcp_tip is not None:
            port_yaw_final = dyaw + self.plug_yaw_offset + tcp_yaw0
            q_final = self._tcp_quat_for_port_yaw(port_yaw_final)
            self.get_logger().info(
                f"ALIGN: tcp orientation set so the plug axis is vertical at port yaw {math.degrees(port_yaw_final):.1f}deg")
        handoff_tip = np.array([entrance[0], entrance[1], entrance[2] + self.handoff_gap_m])
        self.get_logger().info(
            f"HANDOFF: source={source} dyaw={math.degrees(dyaw):.1f} "
            f"tip -> {np.round(handoff_tip, 4).tolist()} entrance={np.round(entrance, 4).tolist()}")
        # hand-off at the start yaw (templates were taught at that camera roll), image servo,
        # then rotate about the tip to the port yaw and insert
        self._move_to(move_robot, get_observation, self._tcp_target_for_tip(handoff_tip, q_tcp))
        self._debug_truth(get_observation, "handoff")
        self._servo_xy(get_observation, move_robot, q_tcp, k)
        self._debug_truth(get_observation, "after_servo")
        tip_now = self._tip_xyz(get_observation()) + np.array([self.handoff_bias_m[0], self.handoff_bias_m[1], 0.0])
        if dyaw != 0.0 or np.any(self.handoff_bias_m):
            self._move_to(move_robot, get_observation, self._tcp_target_for_tip(tip_now, q_final), steps=20)
        self._debug_truth(get_observation, "after_yaw")
        success = self._scripted_insert(get_observation, move_robot)
        self._debug_truth(get_observation, "after_insert")
        return self._finish(success, f"handoff_{source}")

    def _servo_xy(self, get_observation, move_robot, q_final, k):
        """Image-based correction at the hand-off height: move the tip in XY until
        the measured port pixel coincides with the pixel where the entrance must
        appear for the tip to be centred over it."""
        for r in range(self.servo_rounds):
            obs = get_observation()
            cam = self._camera(obs, f"{k}_servo{r}")
            if cam is None:
                return
            rgb, png, K, T_base_cam = cam
            det = self._detector.detect_with_range(rgb, min_score=self.min_box_score)
            if det is None:
                self.get_logger().warning(f"SERVO{r}: no detection")
                return
            tip = self._tip_xyz(obs)
            # the detector reports the port LINK pixel (box centre minus the taught offset);
            # the link sits port_depth below the entrance, which sits handoff_gap below the tip
            want = tip - np.array([0.0, 0.0, self.handoff_gap_m + self.port_depth_m])
            T_cam_base = np.linalg.inv(T_base_cam)
            pc = T_cam_base[:3, :3] @ want + T_cam_base[:3, 3]
            u_des = K @ pc
            u_des = u_des[:2] / u_des[2]
            e = np.array(det["port_px"]) - u_des
            Z = det["range_m"]
            d_cam = np.array([e[0] * Z / K[0, 0], e[1] * Z / K[1, 1], 0.0])
            d_base = T_base_cam[:3, :3] @ d_cam
            self.get_logger().info(
                f"SERVO{r}: port_px={np.round(det['port_px'], 1).tolist()} desired_px={np.round(u_des, 1).tolist()} "
                f"err_px={np.round(e, 1).tolist()} move_xy_mm={np.round(d_base[:2] * 1000, 2).tolist()}")
            if float(np.hypot(*e)) <= self.servo_tol_px:
                return
            if float(np.hypot(*e)) > self.servo_max_px:
                self.get_logger().warning(f"SERVO{r}: error {float(np.hypot(*e)):.0f} px exceeds gate; not moving")
                return
            step = np.clip(d_base[:2], -0.01, 0.01)
            target_tip = np.array([tip[0] + step[0], tip[1] + step[1], tip[2]])
            self._move_to(move_robot, get_observation, self._tcp_target_for_tip(target_tip, q_final), steps=10)

    def _debug_truth(self, get_observation, tag):
        """Log the true tip-to-entrance offset when the port TF exists (ground
        truth on). Diagnostic only: never used for control."""
        task = self._task
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        try:
            T_b_port = _tf_to_mat(self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame, Time()).transform)
            T_b_ent = _tf_to_mat(self._parent_node._tf_buffer.lookup_transform(
                "base_link", port_frame + "_entrance", Time()).transform)
        except TransformException:
            return
        obs = get_observation()
        tip = self._tip_xyz(obs)
        tcp = self._tcp_pose(obs)
        x_axis = T_b_port[:3, 0]
        port_yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
        tcp_yaw = math.degrees(math.atan2(2 * (tcp.orientation.w * tcp.orientation.z + tcp.orientation.x * tcp.orientation.y),
                                          1 - 2 * (tcp.orientation.y ** 2 + tcp.orientation.z ** 2)))
        self.get_logger().info(
            f"TRUTH[{tag}]: tip-entrance_mm={np.round((tip - T_b_ent[:3, 3]) * 1000, 1).tolist()} "
            f"tip-port_mm={np.round((tip - T_b_port[:3, 3]) * 1000, 1).tolist()} "
            f"port_yaw={port_yaw:.1f} tcp_yaw={tcp_yaw:.1f} port_base={np.round(T_b_port[:3, 3], 4).tolist()}")

    def _finish(self, success, reason):
        success = bool(success or self._insertion_event_seen)
        msg = f"megapose_guided reason={reason} event={self._last_event!r}"
        self.get_logger().info(f"FINISH success={success} {msg}")
        result = InsertCable.Result()
        result.success = success
        result.message = msg
        self._task_result_pub.publish(result)
        self._episode_pub.publish(String(data="stop"))
        self.sleep_for(self._inter_trial_wait_s)
        return success

    # ---- scripted insert (port of RunACTStagedScriptedInsert v4) -----------

    def _force_n(self, observation):
        if self._latest_wrench is None:
            return None
        tare = observation.controller_state.fts_tare_offset.wrench.force
        f = self._latest_wrench.wrench.force
        return float(math.sqrt((f.x - tare.x) ** 2 + (f.y - tare.y) ** 2 + (f.z - tare.z) ** 2))

    def _obs_age_s(self, observation):
        stamp = Time.from_msg(observation.controller_state.header.stamp)
        return (self.get_clock().now() - stamp).nanoseconds / 1e9

    def _scripted_insert(self, get_observation, move_robot):
        observation = get_observation()
        start = self._tcp_pose(observation)
        start_xyz = np.array([start.position.x, start.position.y, start.position.z])
        self.get_logger().info(
            f"Scripted insert hand-off at xyz={np.round(start_xyz, 4).tolist()} "
            f"depth={self.insert_depth_m} speed={self.insert_speed_mps} stiffness={self.insert_stiffness}")
        floor_z = start_xyz[2] - self.insert_depth_m
        anchor = start_xyz.copy()
        wiggle_m = self.wiggle_start_m
        best_z = min_z = start_xyz[2]
        last_gain_t = t0 = attempt_t0 = stale_since = time.time()
        history = collections.deque()
        retries, retreat_until, retreat_z = 0, None, None
        reason = "time_limit"
        xy_bias = np.zeros(2)
        while (now := time.time()) - t0 < self.insert_time_s:
            if self._insertion_event_seen:
                reason = "insertion_event"
                break
            observation = get_observation()
            if observation is None:
                continue
            if self._obs_age_s(observation) <= 0.5:
                stale_since = now
            elif now - stale_since > 1.0:
                reason = "stale_observation"
                break
            tcp = self._tcp_pose(observation).position
            min_z = min(min_z, tcp.z)
            force_n = self._force_n(observation)
            if force_n is not None and force_n >= self.stop_force_n:
                reason = "force_stop"
                break
            if retreat_until is not None:
                if now < retreat_until:
                    target_xyz = (anchor[0], anchor[1], retreat_z)
                else:
                    anchor = np.array([anchor[0], anchor[1], tcp.z])
                    attempt_t0, last_gain_t = now, now
                    best_z, wiggle_m = tcp.z, self.wiggle_start_m
                    history.clear()
                    retreat_until = None
            if force_n is not None and force_n < 3.0:
                xy_bias += self.insert_xy_gain * np.array([anchor[0] - tcp.x, anchor[1] - tcp.y])
                xy_bias = np.clip(xy_bias, -0.006, 0.006)
            if retreat_until is None:
                history.append((now, tcp.z))
                while history and now - history[0][0] > self.stall_window_s:
                    history.popleft()
                target_z = max(floor_z, anchor[2] - self.insert_speed_mps * (now - attempt_t0))
                stalled = (now - attempt_t0 > self.stall_window_s
                           and history[0][1] - tcp.z < self.stall_progress_m
                           and tcp.z - target_z > self.min_lead_m)
                if stalled and retries < self.max_retries:
                    retries += 1
                    dx, dy = _RETRY_DIRECTIONS[(retries - 1) % 4]
                    radius = self.retry_offset_m * (1 + (retries - 1) // 4)
                    anchor = np.array([start_xyz[0] + radius * dx, start_xyz[1] + radius * dy, tcp.z])
                    retreat_z = tcp.z + self.retreat_m
                    retreat_until = now + self.retreat_hold_s
                    self.get_logger().info(
                        f"SI_RETRY {retries}/{self.max_retries} at tcp={_xyz(tcp)} "
                        f"target_z={target_z:.4f} force={force_n}")
                    target_xyz = (anchor[0], anchor[1], retreat_z)
                else:
                    if tcp.z < best_z - 0.0005:
                        best_z, last_gain_t = tcp.z, now
                    elif now - last_gain_t > 1.0:
                        wiggle_m = min(wiggle_m + 0.00025, self.wiggle_max_m)
                        last_gain_t = now
                    phase = 2.0 * math.pi * self.wiggle_hz * (now - attempt_t0)
                    target_xyz = (anchor[0] + wiggle_m * math.cos(phase),
                                  anchor[1] + wiggle_m * math.sin(phase), target_z)
            target = Pose(position=Point(x=float(target_xyz[0] + xy_bias[0]), y=float(target_xyz[1] + xy_bias[1]),
                                         z=float(target_xyz[2])),
                          orientation=start.orientation)
            self.set_pose_target(move_robot=move_robot, pose=target,
                                 stiffness=self.insert_stiffness, damping=self.insert_damping)
            if int((now - t0) / self.period_s) % 10 == 0:
                self.get_logger().info(
                    f"SI_TICK t={now - t0:.2f} req={_xyz(target.position)} tcp={_xyz(tcp)} "
                    f"force={None if force_n is None else round(force_n, 2)}")
            time.sleep(max(0.0, self.period_s - (time.time() - now)))
        hold = self._tcp_pose(get_observation())
        self.set_pose_target(move_robot=move_robot, pose=Pose(position=hold.position, orientation=hold.orientation),
                             stiffness=self._approach_stiffness, damping=self._approach_damping)
        self.get_logger().info(
            f"Scripted insert done. reason={reason} descended={(start_xyz[2] - min_z) * 1000:.2f}mm "
            f"retries={retries} xy_bias_mm={np.round(xy_bias * 1000, 2).tolist()} event={self._insertion_event_seen}")
        return self._insertion_event_seen
