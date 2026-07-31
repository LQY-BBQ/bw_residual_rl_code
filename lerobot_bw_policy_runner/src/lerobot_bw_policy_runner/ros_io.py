"""ROS2 input reader and output/debug publisher for policy inference."""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32, Int8MultiArray

from .camera_stream import CameraStreamStatus, CameraStreamTracker
from .config import AppConfig
from .constants import IMAGE_KEY_PREFIX, OBS_STATE_KEY
from .image_utils import ImageConversionError, ros_image_to_rgb
from .joint_mapping import joint_dict_to_vector, state_from_joint_state


@dataclass(slots=True)
class ImageSourceInfo:
    width: int
    height: int
    encoding: str
    step: int

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        return (self.height, self.width, 3)


@dataclass(slots=True)
class ObservationSample:
    observation_state: np.ndarray
    images: dict[str, np.ndarray]
    control_source: int | None = None
    image_sources: dict[str, ImageSourceInfo] = field(default_factory=dict)

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
        self._camera_tracker = CameraStreamTracker(list(config.robot.input_topics.cameras))
        self._consumed_image_sequences = {
            name: 0 for name in config.robot.input_topics.cameras
        }
        self._last_error: str | None = None
        self._last_wait_reason: str | None = None

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
        self.debug_gripper_class_publisher = self.create_publisher(
            Int8MultiArray, config.robot.output_topics.debug_gripper_residual_class, 10
        )

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
        self.get_logger().info(
            f"Debug gripper class:   {config.robot.output_topics.debug_gripper_residual_class}"
        )

    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            self._latest_state = msg

    def _on_control_source(self, msg: Int32) -> None:
        with self._lock:
            self._latest_control_source = msg

    def _make_image_callback(self, camera_name: str):
        def _callback(msg: Image) -> None:
            with self._lock:
                is_unique = self._camera_tracker.update(
                    camera_name,
                    msg,
                    received_monotonic=time.monotonic(),
                )
                if is_unique:
                    self._latest_images[camera_name] = msg
        return _callback

    def camera_source_errors(self) -> list[str]:
        stream_config = self.config.inference.camera_stream
        if stream_config is None:
            return ["inference.camera_stream is not configured"]
        with self._lock:
            image_msgs = dict(self._latest_images)
        errors: list[str] = []
        for camera_name, expected in stream_config.sources.items():
            msg = image_msgs.get(camera_name)
            if msg is None:
                errors.append(f"{camera_name}: no image received")
                continue
            actual = (int(msg.width), int(msg.height), str(msg.encoding).strip().lower())
            wanted = (expected.width, expected.height, expected.encoding)
            if actual != wanted:
                errors.append(
                    f"{camera_name}: source={actual[0]}x{actual[1]} {actual[2]!r}, "
                    f"expected={wanted[0]}x{wanted[1]} {wanted[2]!r}"
                )
        return errors

    def measure_camera_streams(
        self,
        duration_s: float,
        *,
        spin: bool = True,
    ) -> dict[str, CameraStreamStatus]:
        duration_s = max(float(duration_s), 0.01)
        with self._lock:
            start_counts = self._camera_tracker.counters()
        started = time.monotonic()
        deadline = started + duration_s
        while rclpy.ok() and time.monotonic() < deadline:
            if spin:
                rclpy.spin_once(self, timeout_sec=min(0.05, max(deadline - time.monotonic(), 0.0)))
            else:
                time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
        ended = time.monotonic()
        with self._lock:
            return self._camera_tracker.statuses(
                start_counts,
                duration_s=ended - started,
                now_monotonic=ended,
            )

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

    def wait_for_first_messages(self, timeout_s: float, *, spin: bool = True) -> bool:
        deadline = time.monotonic() + timeout_s
        last_log = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            if spin:
                rclpy.spin_once(self, timeout_sec=0.05)
            else:
                time.sleep(0.05)
            missing = self.missing_inputs()
            if not missing:
                return True
            now = time.monotonic()
            if now - last_log > 1.0:
                self.get_logger().info(f"Waiting for {len(missing)} topic(s): {', '.join(missing)}")
                last_log = now
        return not self.missing_inputs()

    def get_latest_sample(
        self,
        *,
        expected_image_shapes: dict[str, tuple[int, int]] | None = None,
        require_new_images: bool | None = None,
        max_image_age_s: float | None = None,
    ) -> ObservationSample | None:
        stream_config = self.config.inference.camera_stream
        if stream_config is None:
            self._last_error = "inference.camera_stream is not configured"
            return None
        if require_new_images is None:
            require_new_images = stream_config.require_new_frames
        if max_image_age_s is None:
            max_image_age_s = stream_config.max_frame_age_s
        with self._lock:
            state_msg = self._latest_state
            control_msg = self._latest_control_source
            image_msgs = dict(self._latest_images)
            image_sequences = self._camera_tracker.sequences()
            image_received_times = self._camera_tracker.received_times()
        if state_msg is None:
            return None
        if self.config.inference.require_all_cameras:
            for camera_name in self.config.robot.input_topics.cameras:
                if camera_name not in image_msgs:
                    return None
        required_camera_names = list(self.config.robot.input_topics.cameras)
        if require_new_images:
            waiting = [
                name
                for name in required_camera_names
                if image_sequences[name] <= self._consumed_image_sequences[name]
            ]
            if waiting:
                self._last_error = None
                self._last_wait_reason = f"waiting for new image(s): {', '.join(waiting)}"
                return None
        now = time.monotonic()
        stale = {
            name: now - image_received_times[name]
            for name in required_camera_names
            if name in image_msgs and now - image_received_times[name] > float(max_image_age_s)
        }
        if stale:
            detail = ", ".join(f"{name}={age:.3f}s" for name, age in stale.items())
            self._last_error = None
            self._last_wait_reason = f"stale camera image(s): {detail}"
            return None
        try:
            state_dict = state_from_joint_state(state_msg, source_label=f"state:{self.config.robot.input_topics.state}")
            images: dict[str, np.ndarray] = {}
            image_sources: dict[str, ImageSourceInfo] = {}
            for camera_name in self.config.robot.input_topics.cameras:
                if camera_name not in image_msgs:
                    continue
                msg = image_msgs[camera_name]
                image_key = f"{IMAGE_KEY_PREFIX}.{camera_name}"
                source = ImageSourceInfo(
                    width=int(msg.width),
                    height=int(msg.height),
                    encoding=str(msg.encoding),
                    step=int(msg.step),
                )
                configured_source = stream_config.sources[camera_name]
                actual_source = (source.width, source.height, source.encoding.strip().lower())
                expected_configured_source = (
                    configured_source.width,
                    configured_source.height,
                    configured_source.encoding,
                )
                if actual_source != expected_configured_source:
                    raise ImageConversionError(
                        f"Camera {camera_name!r} source={actual_source[0]}x{actual_source[1]} "
                        f"{actual_source[2]!r}, expected={expected_configured_source[0]}x"
                        f"{expected_configured_source[1]} {expected_configured_source[2]!r}"
                    )
                image = ros_image_to_rgb(msg)
                if expected_image_shapes and image_key in expected_image_shapes:
                    expected_height, expected_width = expected_image_shapes[image_key]
                    if image.shape[:2] != (expected_height, expected_width):
                        raise ImageConversionError(
                            f"Camera {camera_name!r} RGB shape={image.shape}, ACT expects "
                            f"exact shape={(expected_height, expected_width, 3)}; "
                            "runtime resizing is disabled"
                        )
                images[camera_name] = image
                image_sources[camera_name] = source
            if require_new_images:
                with self._lock:
                    for camera_name, sequence in image_sequences.items():
                        self._consumed_image_sequences[camera_name] = max(
                            self._consumed_image_sequences[camera_name],
                            sequence,
                        )
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().warning(f"Latest observation is invalid: {exc}")
            return None
        self._last_error = None
        self._last_wait_reason = None
        control_source = int(control_msg.data) if control_msg is not None else None
        return ObservationSample(
            observation_state=joint_dict_to_vector(state_dict),
            images=images,
            control_source=control_source,
            image_sources=image_sources,
        )

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
        gripper_classes: np.ndarray,
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
        class_msg = Int8MultiArray()
        class_msg.data = [int(value) for value in np.asarray(gripper_classes).reshape(2)]
        self.debug_gripper_class_publisher.publish(class_msg)

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_wait_reason(self) -> str | None:
        return self._last_wait_reason


def init_ros(config: AppConfig) -> None:
    if not rclpy.ok():
        rclpy.init()
