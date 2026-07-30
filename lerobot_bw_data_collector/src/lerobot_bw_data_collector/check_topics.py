"""Detailed topic checker for BC/RL collector inputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import rclpy

from .config import default_config_path, load_config
from .ros_reader import BWTopicReader, init_ros


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check BW collector ROS2 input topics.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--robot-sn", required=False)
    parser.add_argument("--mode", choices=["bc", "rl"], default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args(argv)


def _topic_type_map(node: BWTopicReader) -> dict[str, list[str]]:
    return {topic_name: list(topic_types) for topic_name, topic_types in node.get_topic_names_and_types()}


def _print_topic_existence(reader: BWTopicReader) -> bool:
    topics = _topic_type_map(reader)
    expected = {
        reader.config.robot.topics.state: "sensor_msgs/msg/JointState",
        reader.config.robot.topics.arm_action: "sensor_msgs/msg/JointState",
        reader.config.robot.topics.gripper_action: "sensor_msgs/msg/JointState",
        **{topic: "sensor_msgs/msg/Image" for topic in reader.config.cameras.topics.values()},
    }
    if reader.config.dataset.mode == "rl":
        expected.update(
            {
                reader.config.robot.topics.control_source: "std_msgs/msg/Int32",
                reader.config.robot.topics.action_act: "sensor_msgs/msg/JointState",
                reader.config.robot.topics.action_rl_delta: "sensor_msgs/msg/JointState",
                reader.config.robot.topics.action_final: "sensor_msgs/msg/JointState",
            }
        )
    ok = True
    print("Configured topics:")
    for topic, expected_type in expected.items():
        actual_types = topics.get(topic)
        if actual_types is None:
            ok = False
            print(f"  [MISSING] {topic} expected={expected_type}")
        elif expected_type not in actual_types:
            ok = False
            print(f"  [TYPE?]   {topic} actual={actual_types} expected={expected_type}")
        else:
            print(f"  [OK]      {topic} type={expected_type}")
    return ok


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, robot_sn=args.robot_sn, mode=args.mode)
    os.environ["ROS_DOMAIN_ID"] = str(config.ros.domain_id)
    timeout_s = float(args.timeout if args.timeout is not None else config.record.warmup_timeout_s)
    reader: BWTopicReader | None = None
    try:
        init_ros(config)
        reader = BWTopicReader(config)
        graph_deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < graph_deadline:
            rclpy.spin_once(reader, timeout_sec=0.1)
        topic_graph_ok = _print_topic_existence(reader)
        print(f"\nWaiting up to {timeout_s:.1f}s for actual messages...")
        messages_ok = reader.wait_for_first_messages(timeout_s=timeout_s)
        if not messages_ok:
            print("\nNo message received from these required topics:")
            for topic in reader.missing_inputs():
                print(f"  - {topic}")
            return 2
        source_errors = reader.camera_source_errors()
        if source_errors:
            print("\nCamera source contract mismatch:", file=sys.stderr)
            for error in source_errors:
                print(f"  - {error}", file=sys.stderr)
            return 3
        print(f"\nMeasuring unique camera frames for {config.cameras.rate_measurement_s:.1f}s...")
        statuses = reader.measure_camera_streams(config.cameras.rate_measurement_s)
        rates_ok = True
        for camera_name, status in statuses.items():
            passed = (
                status.fps >= config.cameras.minimum_fps
                and status.age_s <= config.cameras.max_frame_age_s
                and status.unstamped_frames == 0
            )
            rates_ok = rates_ok and passed
            marker = "OK" if passed else "FAIL"
            print(
                f"  [{marker}] {camera_name:<16} {status.fps:.2f} FPS "
                f"unique={status.unique_frames} duplicate={status.duplicate_frames} "
                f"unstamped={status.unstamped_frames} age={status.age_s:.3f}s"
            )
        if not rates_ok:
            print(
                f"All cameras must sustain at least {config.cameras.minimum_fps:g} FPS.",
                file=sys.stderr,
            )
            return 4
        sample = None
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(reader, timeout_sec=0.05)
            sample = reader.get_latest_sample(require_new_images=False)
            if sample is not None:
                break
        if sample is None:
            print("\nTopics publish, but a valid sample cannot be assembled.", file=sys.stderr)
            if reader.last_error:
                print(f"Last error: {reader.last_error}", file=sys.stderr)
            return 5
        print("\nMessage content check:")
        print(f"  [OK] observation.state shape={sample.observation_state.shape}")
        print(f"  [OK] action            shape={sample.action.shape}")
        camera_info = reader.camera_image_info()
        for camera_name, image in sample.images.items():
            source = camera_info[camera_name]
            print(
                f"  [OK] {camera_name:<16} "
                f"source={source['width']}x{source['height']} {source['encoding']} step={source['step']} "
                f"-> RGB shape={image.shape} dtype={image.dtype}"
            )
        if config.dataset.mode == "rl":
            print(f"  [OK] control_source={sample.control_source} is_intervention={sample.is_intervention}")
            print(f"  [OK] action.act       shape={sample.action_act.shape}")
            print(f"  [OK] action.rl_delta  shape={sample.action_rl_delta.shape}")
            print(f"  [OK] action.executed  shape={sample.action_executed.shape}")
        if not topic_graph_ok:
            print("\nWarning: message content worked, but ROS graph type check had warnings above.")
        print("\nTopic check passed. You can run scripts/collect.sh now.")
        return 0
    finally:
        if reader is not None:
            reader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
