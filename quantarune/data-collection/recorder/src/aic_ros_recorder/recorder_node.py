import argparse
import os
import signal
import subprocess
import re
import shutil
import time
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from aic_task_interfaces.action import InsertCable
from aic_control_interfaces.msg import ControllerState, JointMotionUpdate, MotionUpdate
from aic_task_interfaces.msg import Task
from aic_ros_recorder.aic_metadata import (
    CAMERA_SHAPES,
    OBSERVATION_FEATURES,
    ROBOT_TYPE,
    TASK_TEXT_FEATURES,
    encode_task_numeric,
    encode_task_text,
    task_msg_to_text,
    task_text_dataset_features,
)
from geometry_msgs.msg import WrenchStamped
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    build_dataset_frame,
    hw_to_dataset_features,
)
from lerobot.utils.constants import ACTION, OBS_STR
from PIL import Image as PILImage
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CameraInfo, Image, JointState as RospyJointState
from std_msgs.msg import String


class AICRosRecorder(Node):
    DATASET_FPS = 20
    ROSBAG_ROOT_ENV = "AIC_RECORDER_ROSBAG_ROOT"
    SINGLE_EPISODE_PER_FILE = True

    def __init__(self, number_trials: int | None = None):
        super().__init__("aic_ros_recorder")
        self.number_trials = number_trials

        dataset_root_env = os.getenv("AIC_RECORDER_PARQUET_ROOT") or os.getenv(
            "AIC_RECORDER_BAG_ROOT"
        )
        self.dataset_root = Path(dataset_root_env or "/tmp/aic_ros_recorder")
        self.rosbag_root = Path(
            os.getenv(self.ROSBAG_ROOT_ENV, "/tmp/aic_ros_recorder_bags")
        )
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.rosbag_root.mkdir(parents=True, exist_ok=True)

        self.experiment = self._next_experiment_index()
        self.dataset_repo_id = f"aic_ros_recorder/experiment_{self.experiment}"
        self.experiment_folder = self.dataset_root / f"experiment_{self.experiment}"
        self.rosbag_experiment_folder = (
            self.rosbag_root / f"experiment_{self.experiment}"
        )
        self.rosbag_experiment_folder.mkdir(parents=True, exist_ok=False)

        self.episode = 0
        self.recording = False
        self._bag_process: Optional[subprocess.Popen] = None
        self._bag_log_file: Optional[IO[str]] = None

        self.latest_controller_state: Optional[ControllerState] = None
        self.latest_joint_state: Optional[RospyJointState] = None
        self.latest_gripper_state: Optional[RospyJointState] = None
        self.latest_wrench: Optional[WrenchStamped] = None
        self.latest_motion_update: Optional[MotionUpdate] = None
        self.latest_joint_motion_update: Optional[JointMotionUpdate] = None
        self.latest_task_msg: Optional[Task] = None
        self.latest_camera_images: dict[str, np.ndarray] = {}
        self.latest_task_result_success = -1
        self.latest_task_result_message = ""
        self._warned_image_encodings: set[str] = set()
        self._pending_result_goal_ids: dict[str, Any] = {}
        self._result_requests_in_flight: set[str] = set()
        self._completed_result_goal_ids: set[str] = set()

        self.dataset_features = self._build_dataset_features()

        self.dataset: Optional[LeRobotDataset] = None
        self._rosbag_topics = [
            "/fts_broadcaster/wrench",
            "/joint_states",
            "/gripper_state",
            "/aic_controller/controller_state",
            "/aic_controller/pose_commands",
            "/aic_controller/joint_commands",
            "/cheatcode/task",
            "/cheatcode/task_result",
            "/cheatcode/episode_status",
            "/insert_cable/_action/status",
        ]
        for cam_key in CAMERA_SHAPES:
            self._rosbag_topics.extend(
                [
                    f"/{cam_key}/image",
                    f"/{cam_key}/camera_info",
                ]
            )

        self._bind_topic_subs()

        self.timer = self.create_timer(1.0 / self.DATASET_FPS, self._timer_cb)
        self._action_result_timer = self.create_timer(
            0.25, self._poll_insert_cable_results
        )

        self.get_logger().info(
            "AIC ROS Recorder started with "
            f"dataset root {self.experiment_folder} and rosbag root {self.rosbag_experiment_folder}"
        )
        if self.SINGLE_EPISODE_PER_FILE:
            self.get_logger().info(
                "Recorder storage mode: one LeRobot parquet data file per episode."
            )
        if self.number_trials is not None:
            self.get_logger().info(
                f"Recorder will auto-shutdown after {self.number_trials} trial(s)"
            )

    def _next_experiment_index(self) -> int:
        pattern = re.compile(r"^experiment_(\d+)$")
        max_index = 0
        for root in (self.dataset_root, self.rosbag_root):
            for path in root.iterdir():
                if not path.is_dir():
                    continue
                match = pattern.match(path.name)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def _build_dataset_features(self) -> dict[str, dict[str, Any]]:
        action_features = {
            "cartesian.linear.x": float,
            "cartesian.linear.y": float,
            "cartesian.linear.z": float,
            "cartesian.angular.x": float,
            "cartesian.angular.y": float,
            "cartesian.angular.z": float,
            "cartesian.pose.position.x": float,
            "cartesian.pose.position.y": float,
            "cartesian.pose.position.z": float,
            "cartesian.pose.orientation.x": float,
            "cartesian.pose.orientation.y": float,
            "cartesian.pose.orientation.z": float,
            "cartesian.pose.orientation.w": float,
            "joint.shoulder_pan_joint": float,
            "joint.shoulder_lift_joint": float,
            "joint.elbow_joint": float,
            "joint.wrist_1_joint": float,
            "joint.wrist_2_joint": float,
            "joint.wrist_3_joint": float,
        }
        task_result_features = {
            "task_result.success": {"dtype": "int64", "shape": (1,), "names": None},
            "task_result.message": {"dtype": "string", "shape": (1,), "names": None},
        }

        return {
            **hw_to_dataset_features(
                OBSERVATION_FEATURES, prefix=OBS_STR, use_video=True
            ),
            **task_text_dataset_features(OBS_STR),
            **hw_to_dataset_features(action_features, prefix=ACTION, use_video=True),
            **task_result_features,
        }

    def _open_or_create_episode_dataset(self) -> LeRobotDataset:
        """Open the experiment dataset for a single episode recording session.

        The recorder intentionally opens and finalizes the LeRobot dataset once per
        episode. When LeRobot reopens an existing dataset, it resumes on the next
        `data/chunk-XXX/file-YYY.parquet`, which gives us an explicit
        one-parquet-file-per-episode layout while keeping LeRobot's native metadata
        and indexing scheme.
        """
        tasks_path = self.experiment_folder / "meta" / "tasks.parquet"
        if tasks_path.exists():
            return LeRobotDataset(
                repo_id=self.dataset_repo_id,
                root=str(self.experiment_folder),
                batch_encoding_size=1,
                vcodec="h264",
            )

        if self.experiment_folder.exists():
            shutil.rmtree(self.experiment_folder)

        return LeRobotDataset.create(
            repo_id=self.dataset_repo_id,
            root=str(self.experiment_folder),
            fps=self.DATASET_FPS,
            robot_type=ROBOT_TYPE,
            features=self.dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=0,
            batch_encoding_size=1,
            vcodec="h264",
        )

    def _latest_episode_file_paths(self) -> tuple[Path | None, Path | None]:
        if self.dataset is None:
            return None, None

        data_path = None
        meta_path = None

        if self.dataset.latest_episode is not None:
            latest_data_ep = self.dataset.latest_episode
            data_path = self.experiment_folder / self.dataset.meta.data_path.format(
                chunk_index=latest_data_ep["data/chunk_index"],
                file_index=latest_data_ep["data/file_index"],
            )

        if self.dataset.meta.latest_episode is not None:
            latest_meta_ep = self.dataset.meta.latest_episode
            meta_path = self.experiment_folder / DEFAULT_EPISODES_PATH.format(
                chunk_index=latest_meta_ep["meta/episodes/chunk_index"][0],
                file_index=latest_meta_ep["meta/episodes/file_index"][0],
            )

        return data_path, meta_path

    def _cleanup_temporary_image_cache(self) -> None:
        images_root = self.experiment_folder / "images"
        if images_root.exists():
            shutil.rmtree(images_root, ignore_errors=True)
            self.get_logger().info(
                f"Removed temporary frame image cache at {images_root}; dataset keeps videos under videos/ only."
            )

    def _ros2_bag_available(self) -> bool:
        ros2_bin = shutil.which("ros2")
        if ros2_bin is None:
            return False
        result = subprocess.run(
            [ros2_bin, "bag", "-h"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0

    def _available_rosbag_topics(self) -> list[str]:
        available_topics = {name for name, _types in self.get_topic_names_and_types()}
        recordable_topics = [
            topic for topic in self._rosbag_topics if topic in available_topics
        ]
        missing_topics = [
            topic for topic in self._rosbag_topics if topic not in available_topics
        ]

        if missing_topics:
            self.get_logger().warning(
                "Skipping unavailable rosbag topics: " + ", ".join(missing_topics)
            )

        return recordable_topics

    def _bind_topic_subs(self) -> None:
        """Subscribe to all relevant topics for AIC observer and command capture."""
        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self._controller_state_cb,
            10,
        )
        self.create_subscription(
            RospyJointState,
            "/joint_states",
            self._joint_states_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RospyJointState,
            "/gripper_state",
            self._gripper_state_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String, "/cheatcode/episode_status", self._episode_status_cb, 10
        )
        self.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            self._wrench_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MotionUpdate, "/aic_controller/pose_commands", self._motion_update_cb, 10
        )
        self.create_subscription(
            JointMotionUpdate,
            "/aic_controller/joint_commands",
            self._joint_motion_update_cb,
            10,
        )
        self.create_subscription(Task, "/cheatcode/task", self._task_cb, 10)
        self.create_subscription(
            InsertCable.Result,
            "/cheatcode/task_result",
            self._task_result_cb,
            10,
        )
        self.create_subscription(
            GoalStatusArray,
            "/insert_cable/_action/status",
            self._insert_cable_status_cb,
            10,
        )
        self._insert_cable_get_result_client = self.create_client(
            InsertCable.Impl.GetResultService,
            "/insert_cable/_action/get_result",
        )
        for cam_key in CAMERA_SHAPES:
            self.create_subscription(
                Image,
                f"/{cam_key}/image",
                lambda msg, cam_key=cam_key: self._camera_cb(cam_key, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo,
                f"/{cam_key}/camera_info",
                self._caminfo_cb,
                qos_profile_sensor_data,
            )

    def _controller_state_cb(self, msg: ControllerState) -> None:
        self.latest_controller_state = msg

    def _joint_states_cb(self, msg: RospyJointState) -> None:
        self.latest_joint_state = msg

    def _gripper_state_cb(self, msg: RospyJointState) -> None:
        self.latest_gripper_state = msg

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        self.latest_wrench = msg

    def _motion_update_cb(self, msg: MotionUpdate) -> None:
        self.latest_motion_update = msg

    def _joint_motion_update_cb(self, msg: JointMotionUpdate) -> None:
        self.latest_joint_motion_update = msg

    def _task_cb(self, msg: Task) -> None:
        self.latest_task_msg = msg
        self.latest_task_result_success = -1
        self.latest_task_result_message = ""

    def _task_result_cb(self, msg: InsertCable.Result) -> None:
        self.latest_task_result_success = int(msg.success)
        self.latest_task_result_message = msg.message

        if self.recording and self.dataset is not None:
            self._record_current_frame()

    def _camera_cb(self, cam_key: str, msg: Image) -> None:
        self.latest_camera_images[cam_key] = self._ros_image_to_numpy(cam_key, msg)

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        pass

    def _insert_cable_status_cb(self, msg: GoalStatusArray) -> None:
        terminal_statuses = {
            GoalStatus.STATUS_SUCCEEDED,
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
        }
        for status in msg.status_list:
            if status.status not in terminal_statuses:
                continue

            goal_key = self._goal_id_to_key(status.goal_info.goal_id)
            if goal_key in self._completed_result_goal_ids:
                continue
            self._pending_result_goal_ids[goal_key] = status.goal_info.goal_id

    def _poll_insert_cable_results(self) -> None:
        if not self._pending_result_goal_ids:
            return
        if not self._insert_cable_get_result_client.service_is_ready():
            return

        for goal_key, goal_id in list(self._pending_result_goal_ids.items()):
            if goal_key in self._result_requests_in_flight:
                continue

            request = InsertCable.Impl.GetResultService.Request()
            request.goal_id = goal_id
            future = self._insert_cable_get_result_client.call_async(request)
            self._result_requests_in_flight.add(goal_key)
            future.add_done_callback(
                lambda future, goal_key=goal_key: self._handle_insert_cable_result(
                    goal_key, future
                )
            )

    def _handle_insert_cable_result(self, goal_key: str, future: Any) -> None:
        self._result_requests_in_flight.discard(goal_key)
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to fetch /insert_cable result for goal {goal_key}: {exc}"
            )
            return

        self._pending_result_goal_ids.pop(goal_key, None)
        self._completed_result_goal_ids.add(goal_key)
        self.latest_task_result_success = int(response.result.success)
        self.latest_task_result_message = response.result.message

        if self.recording and self.dataset is not None:
            self._record_current_frame()

    @staticmethod
    def _goal_id_to_key(goal_id: Any) -> str:
        return bytes(goal_id.uuid).hex()

    def _ros_image_to_numpy(self, cam_key: str, msg: Image) -> np.ndarray:
        target_height, target_width, _ = CAMERA_SHAPES[cam_key]
        fallback = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        if msg.height == 0 or msg.width == 0 or not msg.data:
            return fallback

        channels = msg.step // msg.width if msg.width else 0
        if channels <= 0:
            return fallback

        expected_size = msg.step * msg.height
        buffer = np.frombuffer(msg.data, dtype=np.uint8)
        if buffer.size < expected_size:
            return fallback

        rows = buffer[:expected_size].reshape((msg.height, msg.step))
        image = rows[:, : msg.width * channels].reshape(
            (msg.height, msg.width, channels)
        )

        encoding = msg.encoding.lower()
        if encoding == "rgb8":
            converted = image[:, :, :3]
        elif encoding in {"bgr8", "8uc3"}:
            converted = image[:, :, :3][:, :, ::-1]
        elif encoding == "rgba8":
            converted = image[:, :, :3]
        elif encoding == "bgra8":
            converted = image[:, :, [2, 1, 0]]
        elif encoding in {"mono8", "8uc1"}:
            converted = np.repeat(image[:, :, :1], 3, axis=2)
        else:
            if encoding not in self._warned_image_encodings:
                self._warned_image_encodings.add(encoding)
                self.get_logger().warning(
                    f"Unsupported camera encoding {msg.encoding!r} on {cam_key}; using best-effort conversion"
                )
            if channels == 1:
                converted = np.repeat(image[:, :, :1], 3, axis=2)
            else:
                converted = image[:, :, :3]

        if converted.shape[:2] != (target_height, target_width):
            converted = np.asarray(
                PILImage.fromarray(converted).resize(
                    (target_width, target_height), resample=PILImage.BILINEAR
                )
            )

        return converted.astype(np.uint8, copy=False)

    def _current_task_text(self) -> str:
        if self.latest_task_msg is None:
            return ""
        return task_msg_to_text(self.latest_task_msg)

    def _build_task_text_frame(self) -> dict[str, str]:
        if self.latest_task_msg is None:
            return {f"{OBS_STR}.{key}": "" for key in TASK_TEXT_FEATURES}

        return {
            f"{OBS_STR}.{key}": value
            for key, value in encode_task_text(self.latest_task_msg).items()
        }

    def _episode_status_cb(self, msg: String) -> None:
        status = msg.data.strip().lower()
        if status == "start":
            if self.recording:
                self.get_logger().info(
                    "Episode status 'start' ignored because an episode is already recording"
                )
                return
            self.get_logger().info("Episode status 'start' received")
            self.start_episode()
        elif status == "stop":
            self.get_logger().info("Episode status 'stop' received")
            self.stop_episode()
            self._shutdown_if_trial_limit_reached()
        else:
            self.get_logger().warning(
                f"Unsupported /cheatcode/episode_status value {msg.data!r}; expected 'start' or 'stop'"
            )

    def start_episode(self) -> None:
        """Initialize a new episode and start ROS bag + LeRobot dataset recording.

        Called by a "start" message on `/cheatcode/episode_status`. It sets up
        output folders, starts rosbag2 record subprocess and creates a LeRobot
        dataset in which frames will be appended by the timer callback.
        """
        if self.recording:
            self.get_logger().warning("Already recording; stop first")
            return

        self.dataset = self._open_or_create_episode_dataset()
        self.episode += 1
        bag_folder = (
            self.rosbag_experiment_folder / f"episode_{self.episode}" / "rosbag"
        )
        self.latest_task_result_success = -1
        self.latest_task_result_message = ""
        self._pending_result_goal_ids.clear()
        self._result_requests_in_flight.clear()
        self._completed_result_goal_ids.clear()

        self.recording = True
        if not self._ros2_bag_available():
            self.get_logger().warning(
                "ros2 bag command is not available in this environment. "
                "Continuing with parquet dataset recording only."
            )
            self.get_logger().info(
                f"Episode {self.episode} started without rosbag recording"
            )
            return

        topics = self._available_rosbag_topics()
        if not topics:
            self.get_logger().warning(
                "None of the configured rosbag topics are currently available. "
                "Continuing with parquet dataset recording only."
            )
            self.get_logger().info(
                f"Episode {self.episode} started without rosbag recording"
            )
            return

        cmd = ["ros2", "bag", "record", "-o", str(bag_folder)] + topics
        self.get_logger().info(f"Starting rosbag2 record: {' '.join(cmd)}")

        bag_folder.parent.mkdir(parents=True, exist_ok=True)
        bag_log_path = bag_folder.parent / "rosbag_record.log"
        self._bag_log_file = open(bag_log_path, "w")
        self._bag_process = subprocess.Popen(
            cmd,
            stdout=self._bag_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(0.5)
        if self._bag_process.poll() is not None:
            self._bag_log_file.flush()
            error_details = bag_log_path.read_text().strip()
            self._bag_log_file.close()
            self._bag_log_file = None
            self._bag_process = None
            self.get_logger().warning(
                "ros2 bag record exited immediately; continuing with parquet dataset recording only. "
                f"See {bag_log_path} for details."
            )
            if error_details:
                self.get_logger().warning(error_details)
            self.get_logger().info(
                f"Episode {self.episode} started without rosbag recording"
            )
            return

        self.get_logger().info(
            f"Episode {self.episode} started: recording to {bag_folder}"
        )

    def stop_episode(self) -> None:
        """Stop the current recording and finalize the episode dataset.

        This stops the background rosbag process, writes the LeRobot dataset
        to disk, and resets internal recording state.
        """
        if not self.recording:
            self.get_logger().info("No episode is currently recording")
            return

        if self._bag_process is not None:
            self.get_logger().info("Stopping rosbag2 recording")
            os.killpg(os.getpgid(self._bag_process.pid), signal.SIGINT)
            try:
                self._bag_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.get_logger().warning("rosbag record did not stop, terminating")
                self._bag_process.terminate()
                self._bag_process.wait(timeout=5)
            self._bag_process = None
            if self._bag_log_file is not None:
                self._bag_log_file.close()
                self._bag_log_file = None

        if self.dataset is not None:
            episode_buffer = self.dataset.episode_buffer
            episode_size = (
                episode_buffer.get("size", 0) if episode_buffer is not None else 0
            )
            try:
                if episode_size > 0:
                    self._normalize_scalar_episode_features()
                    self.dataset.save_episode()
                    self.dataset.finalize()
                    self._cleanup_temporary_image_cache()
                    data_path, meta_path = self._latest_episode_file_paths()
                    self.get_logger().info(
                        f"Episode {self.episode} appended to LeRobot dataset in {self.experiment_folder}"
                    )
                    if data_path is not None:
                        self.get_logger().info(
                            f"Episode {self.episode} data parquet: {data_path}"
                        )
                    if meta_path is not None:
                        self.get_logger().info(
                            f"Episode {self.episode} episode metadata parquet: {meta_path}"
                        )
                else:
                    self.dataset.clear_episode_buffer(delete_images=True)
                    self._cleanup_temporary_image_cache()
                    self.get_logger().warning(
                        f"Episode {self.episode} had no frames and was discarded"
                    )
                    if (
                        self.dataset.meta.total_episodes == 0
                        and self.experiment_folder.exists()
                    ):
                        shutil.rmtree(self.experiment_folder)
            finally:
                self._cleanup_temporary_image_cache()
                self.dataset = None

        self.recording = False

    def _shutdown_if_trial_limit_reached(self) -> None:
        if self.number_trials is None:
            return
        if self.recording:
            return
        if self.episode < self.number_trials:
            return

        self.get_logger().info(
            f"Recorded {self.episode} episode(s); reaching configured limit {self.number_trials}. Shutting down."
        )
        if rclpy.ok(context=self.context):
            rclpy.shutdown(context=self.context)

    def _timer_cb(self) -> None:
        """Periodic timer callback that writes a frame to the dataset."""
        if not self.recording or self.dataset is None:
            return

        self._record_current_frame()

    def _normalize_scalar_episode_features(self) -> None:
        if self.dataset is None or self.dataset.episode_buffer is None:
            return

        for key, feature in self.dataset.features.items():
            if feature["shape"] != (1,) or feature["dtype"] in {
                "string",
                "image",
                "video",
            }:
                continue

            values = self.dataset.episode_buffer.get(key)
            if not isinstance(values, list):
                continue

            normalized_values = []
            for value in values:
                if isinstance(value, np.ndarray) and value.size == 1:
                    normalized_values.append(value.item())
                else:
                    normalized_values.append(value)
            self.dataset.episode_buffer[key] = normalized_values

    def _record_current_frame(self) -> None:
        if not self.recording or self.dataset is None:
            return

        observation = self._build_observation_payload()
        action = self._build_action_payload()

        obs_frame = build_dataset_frame(
            self.dataset.features, observation, prefix=OBS_STR
        )
        task_text_frame = self._build_task_text_frame()
        action_frame = build_dataset_frame(self.dataset.features, action, prefix=ACTION)

        frame = {
            **obs_frame,
            **task_text_frame,
            **action_frame,
            "task": self._current_task_text(),
            "task_result.success": np.array(
                [self.latest_task_result_success], dtype=np.int64
            ),
            "task_result.message": self.latest_task_result_message,
        }
        self.dataset.add_frame(frame)

    def _build_observation_payload(self) -> dict[str, Any]:
        # default zeros for all observation scalar features
        obs: dict[str, Any] = {}

        if self.latest_controller_state:
            c = self.latest_controller_state
            obs.update(
                {
                    "tcp_pose.position.x": c.tcp_pose.position.x,
                    "tcp_pose.position.y": c.tcp_pose.position.y,
                    "tcp_pose.position.z": c.tcp_pose.position.z,
                    "tcp_pose.orientation.x": c.tcp_pose.orientation.x,
                    "tcp_pose.orientation.y": c.tcp_pose.orientation.y,
                    "tcp_pose.orientation.z": c.tcp_pose.orientation.z,
                    "tcp_pose.orientation.w": c.tcp_pose.orientation.w,
                    "tcp_velocity.linear.x": c.tcp_velocity.linear.x,
                    "tcp_velocity.linear.y": c.tcp_velocity.linear.y,
                    "tcp_velocity.linear.z": c.tcp_velocity.linear.z,
                    "tcp_velocity.angular.x": c.tcp_velocity.angular.x,
                    "tcp_velocity.angular.y": c.tcp_velocity.angular.y,
                    "tcp_velocity.angular.z": c.tcp_velocity.angular.z,
                    "tcp_error.x": c.tcp_error[0],
                    "tcp_error.y": c.tcp_error[1],
                    "tcp_error.z": c.tcp_error[2],
                    "tcp_error.rx": c.tcp_error[3],
                    "tcp_error.ry": c.tcp_error[4],
                    "tcp_error.rz": c.tcp_error[5],
                    "wrench.force.x": c.fts_tare_offset.wrench.force.x,
                    "wrench.force.y": c.fts_tare_offset.wrench.force.y,
                    "wrench.force.z": c.fts_tare_offset.wrench.force.z,
                    "wrench.torque.x": c.fts_tare_offset.wrench.torque.x,
                    "wrench.torque.y": c.fts_tare_offset.wrench.torque.y,
                    "wrench.torque.z": c.fts_tare_offset.wrench.torque.z,
                }
            )

        if self.latest_joint_state:
            pos = list(self.latest_joint_state.position)
            vel = (
                list(self.latest_joint_state.velocity)
                if self.latest_joint_state.velocity
                else [0.0] * len(pos)
            )
            for i in range(7):
                obs[f"joint_positions.{i}"] = float(pos[i] if i < len(pos) else 0.0)
                obs[f"joint_velocity.{i}"] = float(vel[i] if i < len(vel) else 0.0)

        # fill missing keys from controller obs with zero
        for key, spec in OBSERVATION_FEATURES.items():
            if key not in obs and not isinstance(spec, tuple):
                obs[key] = 0.0

        if self.latest_task_msg is not None:
            obs.update(encode_task_numeric(self.latest_task_msg))

        for cam_key, cam_shape in CAMERA_SHAPES.items():
            obs[cam_key] = self.latest_camera_images.get(
                cam_key, np.zeros(cam_shape, dtype=np.uint8)
            )

        return obs

    def _build_action_payload(self) -> dict[str, Any]:
        action: dict[str, Any] = {
            "cartesian.linear.x": 0.0,
            "cartesian.linear.y": 0.0,
            "cartesian.linear.z": 0.0,
            "cartesian.angular.x": 0.0,
            "cartesian.angular.y": 0.0,
            "cartesian.angular.z": 0.0,
            "cartesian.pose.position.x": 0.0,
            "cartesian.pose.position.y": 0.0,
            "cartesian.pose.position.z": 0.0,
            "cartesian.pose.orientation.x": 0.0,
            "cartesian.pose.orientation.y": 0.0,
            "cartesian.pose.orientation.z": 0.0,
            "cartesian.pose.orientation.w": 0.0,
            "joint.shoulder_pan_joint": 0.0,
            "joint.shoulder_lift_joint": 0.0,
            "joint.elbow_joint": 0.0,
            "joint.wrist_1_joint": 0.0,
            "joint.wrist_2_joint": 0.0,
            "joint.wrist_3_joint": 0.0,
        }

        if self.latest_motion_update is not None:
            twist = self.latest_motion_update.velocity
            pose = self.latest_motion_update.pose
            action.update(
                {
                    "cartesian.linear.x": twist.linear.x,
                    "cartesian.linear.y": twist.linear.y,
                    "cartesian.linear.z": twist.linear.z,
                    "cartesian.angular.x": twist.angular.x,
                    "cartesian.angular.y": twist.angular.y,
                    "cartesian.angular.z": twist.angular.z,
                    "cartesian.pose.position.x": pose.position.x,
                    "cartesian.pose.position.y": pose.position.y,
                    "cartesian.pose.position.z": pose.position.z,
                    "cartesian.pose.orientation.x": pose.orientation.x,
                    "cartesian.pose.orientation.y": pose.orientation.y,
                    "cartesian.pose.orientation.z": pose.orientation.z,
                    "cartesian.pose.orientation.w": pose.orientation.w,
                }
            )

        if self.latest_joint_motion_update is not None:
            joints = list(self.latest_joint_motion_update.target_state.velocities)
            keys = [
                "joint.shoulder_pan_joint",
                "joint.shoulder_lift_joint",
                "joint.elbow_joint",
                "joint.wrist_1_joint",
                "joint.wrist_2_joint",
                "joint.wrist_3_joint",
            ]
            for i, k in enumerate(keys):
                action[k] = float(joints[i] if i < len(joints) else 0.0)

        return action

    def log_postprocessing_reminder(self) -> None:
        dataset_root = self.experiment_folder
        scoring_path = dataset_root / "scoring.yaml"
        self.get_logger().warning(
            "Recorder finished. Remember to run scoring postprocessing once "
            f"{scoring_path} is available."
        )
        self.get_logger().warning(
            "Command: "
            f"pixi run ros2 run aic_ros_recorder aic_ros_postprocess_scoring {dataset_root}"
        )


def parse_cli_args(args=None):
    parser = argparse.ArgumentParser(description="AIC ROS data recorder")
    parser.add_argument(
        "--number-trials",
        type=int,
        default=None,
        help="Auto-shutdown after recording this many episodes/trials.",
    )
    cli_args = remove_ros_args(args=args)[1:]
    parsed = parser.parse_args(cli_args)
    if parsed.number_trials is not None and parsed.number_trials <= 0:
        parser.error("--number-trials must be a positive integer")
    return parsed


def main(args=None):
    cli_args = parse_cli_args(args=args)
    rclpy.init(args=args)
    node = AICRosRecorder(number_trials=cli_args.number_trials)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.get_logger().info("Shutdown requested by user")
    finally:
        node.stop_episode()
        node.log_postprocessing_reminder()
        node.destroy_node()
        if rclpy.ok(context=node.context):
            rclpy.shutdown(context=node.context)


if __name__ == "__main__":
    main()
