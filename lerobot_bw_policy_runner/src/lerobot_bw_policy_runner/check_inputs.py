"""Check policy runner input and output topic configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import rclpy

from .config import default_config_path, load_config
from .ros_io import BWObservationReader, init_ros


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--robot-sn", required=False)
    parser.add_argument("--mode", choices=["act", "act_residual_bc", "act_residual_rl", "act_residual_sac"], default=None)
    parser.add_argument("--timeout", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, robot_sn=args.robot_sn, mode=args.mode)
    os.environ["ROS_DOMAIN_ID"] = str(config.ros.domain_id)
    timeout_s = float(args.timeout if args.timeout is not None else config.inference.warmup_timeout_s)
    reader = None
    try:
        init_ros(config)
        reader = BWObservationReader(config)
        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(reader, timeout_sec=0.1)
        topic_types = {name: types for name, types in reader.get_topic_names_and_types()}
        expected = {config.robot.input_topics.state: "sensor_msgs/msg/JointState", **{topic: "sensor_msgs/msg/Image" for topic in config.robot.input_topics.cameras.values()}}
        if config.robot.input_topics.control_source:
            expected[config.robot.input_topics.control_source] = "std_msgs/msg/Int32"
        ok = True
        print("Configured input topics:")
        for topic, typ in expected.items():
            actual = topic_types.get(topic)
            if actual is None:
                ok = False
                print(f"  [MISSING] {topic} expected={typ}")
            elif typ not in actual:
                ok = False
                print(f"  [TYPE?]   {topic} actual={actual} expected={typ}")
            else:
                print(f"  [OK]      {topic} type={typ}")
        print("\nOutput topics:")
        print(f"  {config.robot.output_topics.arm_action}")
        print(f"  {config.robot.output_topics.gripper_action}")
        print("Debug topics:")
        print(f"  {config.robot.output_topics.debug_action_act}")
        print(f"  {config.robot.output_topics.debug_action_rl_delta}")
        print(f"  {config.robot.output_topics.debug_action_composed}")
        print(f"  {config.robot.output_topics.debug_action_final}")
        print(f"\nWaiting up to {timeout_s:.1f}s for messages...")
        if not reader.wait_for_first_messages(timeout_s):
            print("Missing messages:")
            for topic in reader.missing_inputs():
                print(f"  - {topic}")
            return 2
        sample = reader.get_latest_sample()
        if sample is None:
            print(f"Invalid sample: {reader.last_error}")
            return 3
        print("\nInput check passed.")
        print(f"  state shape={sample.observation_state.shape}")
        for name, image in sample.images.items():
            print(f"  {name}: {image.shape}")
        return 0 if ok else 1
    finally:
        if reader is not None:
            reader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
