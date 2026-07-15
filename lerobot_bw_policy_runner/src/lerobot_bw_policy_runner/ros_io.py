"""ROS2 input reader and output/debug publisher for policy inference."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32

from .config import AppConfig
from .constants import IMAGE_KEY_PREFIX, OBS_STATE_KEY
from .image_utils import resize_rgb_image, ros_image_to_rgb
from .joint_mapping import joint_dict_to_vector, state_from_joint_state


@dataclass(slots=True)
class ObservationSample:
    observation_state: np.ndarray
    images: dict[str, np.ndarray]
    control_source: int | None = None

    def to_lerobot_observation(self) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {OBS_STATE_KEY: np.asarray(self.observation_state, dtype=np.float32)}
        for camera_name, image in self.images.items():
            obs[f"{IMAGE_KEY_PREFIX}.{camera_name}"] = np.ascontiguousarray(image, dtype=np.uint8)
        return obs


class BWObservationReader(Node):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(f"lerobot_bw_policy_runner_{config.robot.robot_sn}")
        self.config = config
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._latest_control_source: Int32 | None = None
        self._latest_images: dict[str, Image] = {}
        self._last_error: str | None = None

        joint_qos = 10
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(JointState, config.robot.input_topics.state, self._on_state, joint_qos)
        if config.robot.input_topics.control_source:
            self.create_subscription(Int32, config.robot.input_topics.control_source, self._on_control_source, joint_qos)
        for camera_name, topic in config.robot.input_topics.cameras.items():
            self.create_subscription(Image, topic, self._make_image_callback(camera_name), image_qos)

        self.arm_publisher = self.create_publisher(JointState, config.robot.output_topics.arm_action, 10)
        self.gripper_publisher = self.create_publisher(JointState, config.robot.output_topics.gripper_action, 10)
        self.debug_act_publisher = self.create_publisher(JointState, config.robot.output_topics.debug_action_act, 10)
        self.debug_delta_publisher = self.create_publisher(JointState, config.robot.output_topics.debug_action_rl_delta, 10)
        self.debug_composed_publisher = self.create_publisher(JointState, config.robot.output_topics.debug_action_composed, 10)
        self.debug_final_publisher = self.create_publisher(JointState, config.robot.output_topics.debug_action_final, 10)

        self.get_logger().info(f"Subscribed state: {config.robot.input_topics.state}")
        if config.robot.input_topics.control_source:
            self.get_logger().info(f"Subscribed control_source: {config.robot.input_topics.control_source}")
        for camera_name, topic in config.robot.input_topics.cameras.items():
            self.get_logger().info(f"Subscribed camera {camera_name}: {topic}")
        self.get_logger().info(f"Policy arm output:     {config.robot.output_topics.arm_action}")
        self.get_logger().info(f"Policy gripper output: {config.robot.output_topics.gripper_action}")
        self.get_logger().info(f"Debug action_act:      {config.robot.output_topics.debug_action_act}")
        self.get_logger().info(f"Debug action_delta:    {config.robot.output_topics.debug_action_rl_delta}")
        self.get_logger().info(f"Debug action_composed: {config.robot.output_topics.debug_action_composed}")
        self.get_logger().info(f"Debug action_final:    {config.robot.output_topics.debug_action_final}")

    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            self._latest_state = msg

    def _on_control_source(self, msg: Int32) -> None:
        with self._lock:
            self._latest_control_source = msg

    def _make_image_callback(self, camera_name: str):
        def _callback(msg: Image) -> None:
            with self._lock:
                self._latest_images[camera_name] = msg
        return _callback

    def missing_inputs(self) -> list[str]:
        with self._lock:
            missing: list[str] = []
            if self._latest_state is None:
                missing.append(self.config.robot.input_topics.state)
            if self.config.inference.require_all_cameras:
                for camera_name, topic in self.config.robot.input_topics.cameras.items():
                    if camera_name not in self._latest_images:
                        missing.append(topic)
            return missing

    def wait_for_first_messages(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            missing = self.missing_inputs()
            if not missing:
                return True
            now = time.monotonic()
            if now - last_log > 1.0:
                self.get_logger().info(f"Waiting for {len(missing)} topic(s): {', '.join(missing)}")
                last_log = now
        return not self.missing_inputs()

    def get_latest_sample(self, *, expected_image_shapes: dict[str, tuple[int, int]] | None = None, resize_images: bool = True) -> ObservationSample | None:
        with self._lock:
            state_msg = self._latest_state
            control_msg = self._latest_control_source
            image_msgs = dict(self._latest_images)
        if state_msg is None:
            return None
        if self.config.inference.require_all_cameras:
            for camera_name in self.config.robot.input_topics.cameras:
                if camera_name not in image_msgs:
                    return None
        try:
            state_dict = state_from_joint_state(state_msg, source_label=f"state:{self.config.robot.input_topics.state}")
            images: dict[str, np.ndarray] = {}
            for camera_name in self.config.robot.input_topics.cameras:
                if camera_name not in image_msgs:
                    continue
                image = ros_image_to_rgb(image_msgs[camera_name])
                image_key = f"{IMAGE_KEY_PREFIX}.{camera_name}"
                if resize_images and expected_image_shapes and image_key in expected_image_shapes:
                    height, width = expected_image_shapes[image_key]
                    image = resize_rgb_image(image, height=height, width=width)
                images[camera_name] = image
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().warning(f"Latest observation is invalid: {exc}")
            return None
        self._last_error = None
        control_source = int(control_msg.data) if control_msg is not None else None
        return ObservationSample(observation_state=joint_dict_to_vector(state_dict), images=images, control_source=control_source)

    def publish_action(self, arm_msg: JointState, gripper_msg: JointState, *, dry_run: bool = False) -> None:
        if dry_run:
            return
        self.arm_publisher.publish(arm_msg)
        self.gripper_publisher.publish(gripper_msg)

    def publish_debug_actions(
        self,
        act_msg: JointState,
        delta_msg: JointState,
        composed_msg: JointState,
        final_msg: JointState,
        *,
        dry_run: bool = False,
    ) -> None:
        # Debug topics are safe: they are not consumed by mantis_comm_node as control inputs.
        if dry_run:
            return
        self.debug_act_publisher.publish(act_msg)
        self.debug_delta_publisher.publish(delta_msg)
        self.debug_composed_publisher.publish(composed_msg)
        self.debug_final_publisher.publish(final_msg)

    @property
    def last_error(self) -> str | None:
        return self._last_error


def init_ros(config: AppConfig) -> None:
    if not rclpy.ok():
        rclpy.init()
