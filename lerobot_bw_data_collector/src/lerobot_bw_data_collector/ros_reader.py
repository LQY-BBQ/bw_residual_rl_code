"""ROS2 topic reader for BW BC/RL data collection."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Int32

from .camera_stream import CameraStreamStatus, CameraStreamTracker
from .config import AppConfig
from .image_utils import ros_image_to_rgb
from .joint_mapping import action_from_joint_states, joint_dict_to_vector, state_from_joint_state, vector_from_joint_state


@dataclass(slots=True)
class CollectorSample:
    observation_state: Any
    action: Any
    images: dict[str, Any]
    control_source: int | None = None
    is_intervention: bool = False
    has_human_action: bool = False
    action_act: Any | None = None
    action_rl_delta: Any | None = None
    action_human: Any | None = None
    action_executed: Any | None = None
    timing: dict[str, float] | None = None


class BWTopicReader(Node):
    """Subscribe to robot state/action/camera/debug topics.

    BC mode reads Teleop action as the dataset action.
    RL mode also reads policy debug actions and control_source to form transitions.
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__(f"lerobot_bw_data_collector_{config.robot.robot_sn}_{config.dataset.mode}")
        self.config = config
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._latest_arm_action: JointState | None = None
        self._latest_gripper_action: JointState | None = None
        self._latest_control_source: Int32 | None = None
        self._latest_action_act: JointState | None = None
        self._latest_action_rl_delta: JointState | None = None
        self._latest_action_final: JointState | None = None
        self._latest_images: dict[str, Image] = {}
        self._camera_tracker = CameraStreamTracker(list(config.cameras.topics))
        self._consumed_image_sequences = {name: 0 for name in config.cameras.topics}
        self._last_error: str | None = None
        self._last_wait_reason: str | None = None

        joint_qos = 10
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(JointState, config.robot.topics.state, self._on_state, joint_qos)
        self.create_subscription(JointState, config.robot.topics.arm_action, self._on_arm_action, joint_qos)
        self.create_subscription(JointState, config.robot.topics.gripper_action, self._on_gripper_action, joint_qos)
        if config.dataset.mode == "rl":
            self.create_subscription(Int32, config.robot.topics.control_source, self._on_control_source, joint_qos)
            self.create_subscription(JointState, config.robot.topics.action_act, self._on_action_act, joint_qos)
            self.create_subscription(JointState, config.robot.topics.action_rl_delta, self._on_action_rl_delta, joint_qos)
            self.create_subscription(JointState, config.robot.topics.action_final, self._on_action_final, joint_qos)

        for camera_name, topic in config.cameras.topics.items():
            self.create_subscription(Image, topic, self._make_image_callback(camera_name), image_qos)

        self.get_logger().info(f"Subscribed state:       {config.robot.topics.state}")
        self.get_logger().info(f"Subscribed Teleop arm:  {config.robot.topics.arm_action}")
        self.get_logger().info(f"Subscribed Teleop grip: {config.robot.topics.gripper_action}")
        if config.dataset.mode == "rl":
            self.get_logger().info(f"Subscribed control:     {config.robot.topics.control_source}")
            self.get_logger().info(f"Subscribed action_act:  {config.robot.topics.action_act}")
            self.get_logger().info(f"Subscribed rl_delta:    {config.robot.topics.action_rl_delta}")
            self.get_logger().info(f"Subscribed action_final:{config.robot.topics.action_final}")
        for camera_name, topic in config.cameras.topics.items():
            self.get_logger().info(f"Subscribed camera {camera_name}: {topic}")

    def _on_state(self, msg: JointState) -> None:
        with self._lock: self._latest_state = msg
    def _on_arm_action(self, msg: JointState) -> None:
        with self._lock: self._latest_arm_action = msg
    def _on_gripper_action(self, msg: JointState) -> None:
        with self._lock: self._latest_gripper_action = msg
    def _on_control_source(self, msg: Int32) -> None:
        with self._lock: self._latest_control_source = msg
    def _on_action_act(self, msg: JointState) -> None:
        with self._lock: self._latest_action_act = msg
    def _on_action_rl_delta(self, msg: JointState) -> None:
        with self._lock: self._latest_action_rl_delta = msg
    def _on_action_final(self, msg: JointState) -> None:
        with self._lock: self._latest_action_final = msg

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

    def missing_inputs(self) -> list[str]:
        with self._lock:
            missing: list[str] = []
            if self._latest_state is None: missing.append(self.config.robot.topics.state)
            if self._latest_arm_action is None: missing.append(self.config.robot.topics.arm_action)
            if self._latest_gripper_action is None: missing.append(self.config.robot.topics.gripper_action)
            for camera_name, topic in self.config.cameras.topics.items():
                if camera_name not in self._latest_images: missing.append(topic)
            if self.config.dataset.mode == "rl" and self.config.record.require_rl_debug_topics:
                if self._latest_control_source is None: missing.append(self.config.robot.topics.control_source or "<control_source>")
                if self._latest_action_act is None: missing.append(self.config.robot.topics.action_act or "<action_act>")
                if self._latest_action_rl_delta is None: missing.append(self.config.robot.topics.action_rl_delta or "<action_rl_delta>")
                if self._latest_action_final is None: missing.append(self.config.robot.topics.action_final or "<action_final>")
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

    def camera_image_info(self) -> dict[str, dict[str, int | str]]:
        """Return source metadata for the latest message from each camera."""
        with self._lock:
            image_msgs = dict(self._latest_images)
        return {
            camera_name: {
                "topic": self.config.cameras.topics[camera_name],
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": str(msg.encoding),
                "step": int(msg.step),
            }
            for camera_name, msg in image_msgs.items()
        }

    def camera_source_errors(self) -> list[str]:
        """Validate the current ROS messages against the configured camera contract."""
        with self._lock:
            image_msgs = dict(self._latest_images)
        errors: list[str] = []
        for camera_name, expected in self.config.cameras.sources.items():
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
        """Measure unique, timestamp-deduplicated camera frames over a wall-clock window."""
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

    @staticmethod
    def _stamp_to_float(msg: object | None) -> float | None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None:
            return None
        return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9

    def get_latest_sample(
        self,
        *,
        require_new_images: bool | None = None,
        max_image_age_s: float | None = None,
    ) -> CollectorSample | None:
        if require_new_images is None:
            require_new_images = self.config.cameras.require_new_frames
        if max_image_age_s is None:
            max_image_age_s = self.config.cameras.max_frame_age_s
        with self._lock:
            state_msg = self._latest_state
            arm_msg = self._latest_arm_action
            gripper_msg = self._latest_gripper_action
            control_msg = self._latest_control_source
            act_msg = self._latest_action_act
            delta_msg = self._latest_action_rl_delta
            final_msg = self._latest_action_final
            image_msgs = dict(self._latest_images)
            image_sequences = self._camera_tracker.sequences()
            image_received_times = self._camera_tracker.received_times()

        if state_msg is None or arm_msg is None or gripper_msg is None:
            return None
        for camera_name in self.config.cameras.topics:
            if camera_name not in image_msgs:
                return None
        if require_new_images:
            waiting = [
                name
                for name in self.config.cameras.topics
                if image_sequences[name] <= self._consumed_image_sequences[name]
            ]
            if waiting:
                self._last_error = None
                self._last_wait_reason = f"waiting for new image(s): {', '.join(waiting)}"
                return None
        now = time.monotonic()
        stale = {
            name: now - image_received_times[name]
            for name in self.config.cameras.topics
            if now - image_received_times[name] > float(max_image_age_s)
        }
        if stale:
            detail = ", ".join(f"{name}={age:.3f}s" for name, age in stale.items())
            self._last_error = None
            self._last_wait_reason = f"stale camera image(s): {detail}"
            return None
        if self.config.dataset.mode == "rl" and self.config.record.require_rl_debug_topics:
            if control_msg is None or act_msg is None or delta_msg is None or final_msg is None:
                return None

        try:
            source_errors = self.camera_source_errors()
            if source_errors:
                raise ValueError("; ".join(source_errors))
            state_dict = state_from_joint_state(state_msg, source_label=f"state:{self.config.robot.topics.state}")
            human_action_dict = action_from_joint_states(
                arm_msg, gripper_msg,
                arm_source_label=f"arm_action:{self.config.robot.topics.arm_action}",
                gripper_source_label=f"gripper_action:{self.config.robot.topics.gripper_action}",
            )
            state_vec = joint_dict_to_vector(state_dict)
            human_vec = joint_dict_to_vector(human_action_dict)
            images = {camera_name: ros_image_to_rgb(image_msgs[camera_name]) for camera_name in self.config.cameras.topics}

            if require_new_images:
                with self._lock:
                    for camera_name, sequence in image_sequences.items():
                        self._consumed_image_sequences[camera_name] = max(
                            self._consumed_image_sequences[camera_name],
                            sequence,
                        )

            if self.config.dataset.mode == "bc":
                self._last_error = None
                self._last_wait_reason = None
                return CollectorSample(observation_state=state_vec, action=human_vec, images=images)

            control_source = int(getattr(control_msg, "data", 0)) if control_msg is not None else -1
            is_intervention = control_source == 0
            act_vec = vector_from_joint_state(act_msg, source_label="Policy/debug/action_act") if act_msg is not None else np.zeros_like(human_vec)
            delta_vec = vector_from_joint_state(delta_msg, source_label="Policy/debug/action_rl_delta") if delta_msg is not None else np.zeros_like(human_vec)
            final_vec = vector_from_joint_state(final_msg, source_label="Policy/debug/action_final") if final_msg is not None else np.zeros_like(human_vec)
            action_human = human_vec if is_intervention else np.zeros_like(human_vec)
            action_executed = human_vec if is_intervention else final_vec
            timing = {}
            state_t = self._stamp_to_float(state_msg)
            if state_t is not None:
                for name, msg in [("arm_action_dt", arm_msg), ("gripper_action_dt", gripper_msg), ("action_act_dt", act_msg), ("action_final_dt", final_msg)]:
                    msg_t = self._stamp_to_float(msg)
                    if msg_t is not None:
                        timing[name] = float(msg_t - state_t)
        except Exception as exc:
            self._last_error = str(exc)
            self.get_logger().warning(f"Latest sample is invalid: {exc}")
            return None

        self._last_error = None
        self._last_wait_reason = None
        return CollectorSample(
            observation_state=state_vec,
            action=action_executed,
            images=images,
            control_source=control_source,
            is_intervention=is_intervention,
            has_human_action=is_intervention,
            action_act=act_vec,
            action_rl_delta=delta_vec,
            action_human=action_human,
            action_executed=action_executed,
            timing=timing,
        )

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def last_wait_reason(self) -> str | None:
        return self._last_wait_reason


def init_ros(config: AppConfig) -> None:
    if not rclpy.ok():
        rclpy.init()
