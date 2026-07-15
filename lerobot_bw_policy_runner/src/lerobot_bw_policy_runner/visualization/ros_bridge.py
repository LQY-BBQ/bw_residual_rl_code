"""ROS2 subscriber bridge for the action visualization process."""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from ..config import OutputTopics
from .buffer import ActionHistory, ActionStream


class ActionDebugSubscriber(Node):
    def __init__(self, *, robot_sn: str, topics: OutputTopics, history: ActionHistory) -> None:
        super().__init__(f"lerobot_bw_action_visualizer_{robot_sn}")
        self._history = history
        self._last_warning_time = 0.0
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        stream_topics: tuple[tuple[ActionStream, str], ...] = (
            ("act", topics.debug_action_act),
            ("delta", topics.debug_action_rl_delta),
            ("composed", topics.debug_action_composed),
            ("final", topics.debug_action_final),
        )
        self._subscriptions = [
            self.create_subscription(JointState, topic, self._callback_for(stream), qos)
            for stream, topic in stream_topics
        ]
        for stream, topic in stream_topics:
            self.get_logger().info(f"Subscribed {stream:8s}: {topic}")

    def _callback_for(self, stream: ActionStream):
        def callback(msg: JointState) -> None:
            stamp = msg.header.stamp
            timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            try:
                self._history.add_message(stream, timestamp_ns, msg.name, msg.position)
            except ValueError as exc:
                now = time.monotonic()
                if now - self._last_warning_time >= 2.0:
                    self.get_logger().warning(f"Ignoring invalid {stream} debug message: {exc}")
                    self._last_warning_time = now

        return callback


class VisualizationRosRunner:
    """Own the ROS node and executor on a background thread."""

    def __init__(self, *, robot_sn: str, topics: OutputTopics, history: ActionHistory) -> None:
        self.robot_sn = robot_sn
        self.topics = topics
        self.history = history
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._node: ActionDebugSubscriber | None = None
        self._error_lock = threading.Lock()
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = ActionDebugSubscriber(robot_sn=self.robot_sn, topics=self.topics, history=self.history)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, name="action-visualizer-ros", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._executor is not None:
            self._executor.wake()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        if self._thread is not None and self._thread.is_alive() and rclpy.ok():
            rclpy.shutdown()
            self._thread.join(timeout=1.0)

    @property
    def last_error(self) -> str | None:
        with self._error_lock:
            return self._last_error

    def _spin(self) -> None:
        assert self._executor is not None
        try:
            while not self._stop_event.is_set() and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as exc:
            with self._error_lock:
                self._last_error = str(exc)
        finally:
            if self._node is not None:
                self._executor.remove_node(self._node)
                self._node.destroy_node()
                self._node = None
            self._executor.shutdown(timeout_sec=0.5)
            if rclpy.ok():
                rclpy.shutdown()
