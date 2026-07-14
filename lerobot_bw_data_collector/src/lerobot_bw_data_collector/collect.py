"""Main collection entry point. Use scripts/collect.sh normally."""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import rclpy

from .config import default_config_path, load_config
from .dataset_writer import build_frame, create_lerobot_dataset, finalize_dataset
from .keyboard_marker import KeyboardRewardMarker, MarkerDecision
from .ros_reader import BWTopicReader, init_ros

_SHOULD_STOP = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001
    del signum, frame
    global _SHOULD_STOP
    _SHOULD_STOP = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BW ROS2 topics into a LeRobot dataset.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--robot-sn", required=False)
    parser.add_argument("--task", required=False)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["bc", "rl"],
        default=None,
        help="bc keeps old behavior; rl records residual-SAC transition fields and keyboard rewards.",
    )
    parser.add_argument("--episode-type", default=None, help="demo, correction, rollout, eval.")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means record until Ctrl+C / RL keyboard stop.")
    return parser.parse_args(argv)


def _add_pending_frame(
    dataset,  # noqa: ANN001
    pending_sample,  # noqa: ANN001
    config,  # noqa: ANN001
    decision: MarkerDecision,
) -> None:
    dataset.add_frame(
        build_frame(
            pending_sample,
            config.dataset.task,
            mode=config.dataset.mode,
            reward=decision.reward,
            done=decision.done,
            success=decision.success,
        )
    )


def main(argv: list[str] | None = None) -> int:
    global _SHOULD_STOP
    args = parse_args(argv)
    config = load_config(
        args.config,
        robot_sn=args.robot_sn,
        dataset_root=args.dataset_root,
        task=args.task,
        fps=args.fps,
        mode=args.mode,
        episode_type=args.episode_type,
    )
    os.environ["ROS_DOMAIN_ID"] = str(config.ros.domain_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    reader: BWTopicReader | None = None
    dataset = None
    dataset_path = None
    frame_count = 0
    target_dt = 1.0 / max(float(config.dataset.fps), 1.0)
    pending_sample = None
    pending_decision = MarkerDecision()
    stop_reason = "ctrl_c_or_signal"

    try:
        init_ros(config)
        reader = BWTopicReader(config)
        print(f"\n[1/3] Waiting for topics. mode={config.dataset.mode}, episode_type={config.dataset.episode_type}")
        ready = reader.wait_for_first_messages(timeout_s=config.record.warmup_timeout_s)
        if not ready and config.record.require_all_topics:
            print("ERROR: timed out waiting for required topics:", file=sys.stderr)
            for topic in reader.missing_inputs():
                print(f"  - {topic}", file=sys.stderr)
            return 2

        first_sample = None
        deadline = time.monotonic() + max(config.record.warmup_timeout_s, 1.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(reader, timeout_sec=0.05)
            first_sample = reader.get_latest_sample()
            if first_sample is not None:
                break
        if first_sample is None:
            print("ERROR: no complete valid sample could be assembled.", file=sys.stderr)
            if reader.last_error:
                print(f"Last error: {reader.last_error}", file=sys.stderr)
            return 3

        print("[2/3] Creating LeRobot dataset...")
        handle = create_lerobot_dataset(config, first_sample, session_name=args.session_name, overwrite=args.overwrite)
        dataset = handle.dataset
        dataset_path = handle.dataset_path
        print(f"Dataset path: {dataset_path}")
        print(f"Repo id:      {handle.repo_id}")
        print("Features:")
        for key, value in handle.features.items():
            print(f"  - {key}: shape={value['shape']} dtype={value['dtype']}")

        print("\n[3/3] Recording.")
        marker_enabled = config.dataset.mode == "rl"
        with KeyboardRewardMarker(enable=marker_enabled) as marker:
            if config.dataset.mode == "rl":
                print(marker.help_text())
            else:
                print("BC mode: Press Ctrl+C to stop and save the episode.")

            next_time = time.monotonic()
            last_log_time = time.monotonic()
            while rclpy.ok() and not _SHOULD_STOP:
                rclpy.spin_once(reader, timeout_sec=0.001)
                now = time.monotonic()
                if now < next_time:
                    time.sleep(min(next_time - now, 0.002))
                    continue
                next_time += target_dt

                sample = reader.get_latest_sample()
                if sample is None:
                    if reader.last_error:
                        print(f"Skipping invalid sample: {reader.last_error}", file=sys.stderr)
                    continue

                # Save the previous sample first. The new sample will become the
                # pending frame, so any key pressed in this cycle labels the new
                # frame at index == frame_count after this block.
                if pending_sample is not None:
                    _add_pending_frame(dataset, pending_sample, config, pending_decision)
                    frame_count += 1

                current_frame_index = frame_count
                decision = marker.poll(frame_index=current_frame_index) if config.dataset.mode == "rl" else MarkerDecision()
                pending_sample = sample
                pending_decision = decision

                if decision.done:
                    stop_reason = decision.stop_reason or "keyboard_done"
                    print(f"Stopping episode: {stop_reason}")
                    break

                if config.record.log_every_n_frames > 0 and frame_count > 0 and frame_count % config.record.log_every_n_frames == 0:
                    elapsed = max(time.monotonic() - last_log_time, 1e-6)
                    print(f"Recorded {frame_count} frames ({config.record.log_every_n_frames / elapsed:.1f} fps recent)")
                    last_log_time = time.monotonic()

                if args.max_frames > 0 and (frame_count + 1) >= args.max_frames:
                    print(f"Reached --max-frames {args.max_frames}; saving episode as failure/timeout.")
                    pending_decision.done = True
                    pending_decision.success = False
                    stop_reason = "max_frames"
                    break

        if pending_sample is not None:
            # If the episode stopped via Ctrl+C, SIGTERM, or max frame count, the
            # last saved frame still needs done=True. Keyboard stop already set it.
            if not pending_decision.done:
                pending_decision.done = True
                pending_decision.success = False
            _add_pending_frame(dataset, pending_sample, config, pending_decision)
            frame_count += 1

        if frame_count <= 0:
            print("No frames recorded; nothing to save.")
            return 4

        print(f"Saving episode with {frame_count} frames... stop_reason={stop_reason}")
        dataset.save_episode()
        finalize_dataset(dataset)
        print(f"Saved dataset episode to: {dataset_path}")
        if config.dataset.mode == "rl":
            print("RL rewards were written directly into the dataset 'reward' column; no annotation JSON is required.")
        return 0

    except KeyboardInterrupt:
        _SHOULD_STOP = True
        if dataset is not None and pending_sample is not None:
            if not pending_decision.done:
                pending_decision.done = True
                pending_decision.success = False
            _add_pending_frame(dataset, pending_sample, config, pending_decision)
            frame_count += 1
            print(f"\nSaving episode with {frame_count} frames... stop_reason=keyboard_interrupt")
            dataset.save_episode()
            finalize_dataset(dataset)
            print(f"Saved dataset episode to: {dataset_path}")
            return 0
        print("Interrupted before any frames were recorded.")
        return 130
    finally:
        if reader is not None:
            reader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
