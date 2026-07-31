"""ROS2 deployment for ACT, ACT+residual BC, and ACT+residual RL.

For residual modes, ACT action and pooled ACT visual features are produced by a
single fresh ACT forward on every control cycle.  ACT temporal ensembling is
preserved when it is enabled in the checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor

from .action_utils import (
    ActionCSVLogger,
    ActionFilterState,
    align_action_vector,
    apply_clamp,
    apply_smoothing,
    compose_residual_action,
    split_action_to_joint_states,
    vector_to_joint_state,
)
from .config import default_config_path, load_config
from .constants import (
    GRIPPER_JOINT_INDICES,
    BW_IMAGE_HWC_SHAPES,
    CAMERA_CONTRACT_VERSION,
    IMAGE_TRANSFORM,
    JOINT_NAMES,
)
from .gripper_control import BinaryGripperController
from .policy_loader import (
    infer_action,
    infer_action_with_shared_visual_feature,
    load_policy_bundle,
)
from .residual_policy import build_residual_runtime_obs, infer_residual_action, load_residual_policy
from .ros_io import BWObservationReader, init_ros

_SHOULD_STOP = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001, ARG001
    global _SHOULD_STOP
    _SHOULD_STOP = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACT, ACT+residual BC, or ACT+residual RL on BW.")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--robot-sn", required=False)
    parser.add_argument(
        "--mode",
        choices=["act", "act_residual_bc", "act_residual_rl", "act_residual_sac"],
        default=None,
        help="act_residual_sac is retained as an alias of act_residual_rl.",
    )
    parser.add_argument("--policy-path", required=False, help="ACT LeRobot pretrained_model/checkpoint/output dir.")
    parser.add_argument("--residual-policy-path", required=False, help="Residual BC or RL checkpoint/output dir.")
    parser.add_argument("--residual-lambda", type=float, default=None)
    parser.add_argument("--device", required=False)
    parser.add_argument("--fps", type=float, required=False)
    parser.add_argument("--task", required=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=None)
    hysteresis = parser.add_mutually_exclusive_group()
    hysteresis.add_argument("--gripper-hysteresis", dest="gripper_hysteresis", action="store_true")
    hysteresis.add_argument("--no-gripper-hysteresis", dest="gripper_hysteresis", action="store_false")
    parser.set_defaults(gripper_hysteresis=None)
    return parser.parse_args(argv)


def _validate_residual_pair(mode: str, act_bundle, residual_bundle) -> None:  # noqa: ANN001
    required_type = "residual_bc" if mode == "act_residual_bc" else "residual_rl"
    if residual_bundle.policy_type != required_type:
        raise ValueError(
            f"Mode {mode} requires policy_type={required_type}, checkpoint contains {residual_bundle.policy_type}"
        )
    if not residual_bundle.act_fingerprint:
        raise ValueError("Residual checkpoint has no ACT fingerprint")
    if residual_bundle.act_fingerprint != act_bundle.fingerprint:
        raise ValueError(
            "Residual checkpoint was trained with a different ACT checkpoint. "
            "Use the exact ACT pretrained_model used to build its visual cache."
        )
    if tuple(residual_bundle.image_keys) != tuple(act_bundle.image_keys):
        raise ValueError("Residual checkpoint camera keys/order do not match the ACT checkpoint")
    if residual_bundle.visual_feature_dim != act_bundle.visual_feature_dim:
        raise ValueError(
            f"Residual visual_feature_dim={residual_bundle.visual_feature_dim}, "
            f"ACT produces {act_bundle.visual_feature_dim}"
        )
    if residual_bundle.source_image_shapes != BW_IMAGE_HWC_SHAPES:
        raise ValueError(
            "Residual checkpoint source image shapes do not match the third-generation BW contract: "
            f"checkpoint={residual_bundle.source_image_shapes}, expected={BW_IMAGE_HWC_SHAPES}"
        )
    if residual_bundle.policy_image_shapes != act_bundle.image_shapes:
        raise ValueError(
            "Residual checkpoint ACT image shapes do not match the loaded ACT checkpoint: "
            f"residual={residual_bundle.policy_image_shapes}, ACT={act_bundle.image_shapes}"
        )
    if residual_bundle.camera_contract_version != CAMERA_CONTRACT_VERSION:
        raise ValueError(
            "Residual checkpoint camera contract version does not match this runner: "
            f"checkpoint={residual_bundle.camera_contract_version}, runner={CAMERA_CONTRACT_VERSION}"
        )
    if residual_bundle.image_transform != IMAGE_TRANSFORM:
        raise ValueError(
            "Residual checkpoint image transform does not match this runner: "
            f"checkpoint={residual_bundle.image_transform!r}, runner={IMAGE_TRANSFORM!r}"
        )
    dataset_fps = getattr(residual_bundle, "dataset_fps", None)
    if dataset_fps is not None and not np.isclose(dataset_fps, 30.0, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Residual checkpoint was trained at {dataset_fps:g} FPS; BW runtime requires 30 FPS"
        )


def _camera_streams_ready(reader: BWObservationReader, *, spin: bool) -> bool:
    stream_config = reader.config.inference.camera_stream
    if stream_config is None:
        print("ERROR: inference.camera_stream is not configured.", file=sys.stderr)
        return False
    source_errors = reader.camera_source_errors()
    if source_errors:
        print("ERROR: camera source contract mismatch:", file=sys.stderr)
        for error in source_errors:
            print(f"  - {error}", file=sys.stderr)
        return False
    print(f"Measuring unique camera frames for {stream_config.rate_measurement_s:.1f}s...")
    statuses = reader.measure_camera_streams(stream_config.rate_measurement_s, spin=spin)
    ok = True
    for camera_name, status in statuses.items():
        passed = (
            status.fps >= stream_config.minimum_fps
            and status.age_s <= stream_config.max_frame_age_s
            and status.unstamped_frames == 0
        )
        ok = ok and passed
        marker = "OK" if passed else "FAIL"
        print(
            f"  [{marker}] {camera_name:<16} {status.fps:.2f} FPS "
            f"unique={status.unique_frames} duplicate={status.duplicate_frames} "
            f"unstamped={status.unstamped_frames} age={status.age_s:.3f}s"
        )
    if not ok:
        print(
            f"ERROR: all cameras must sustain at least {stream_config.minimum_fps:g} FPS "
            f"with frame age <= {stream_config.max_frame_age_s:g}s.",
            file=sys.stderr,
        )
    return ok


def main(argv: list[str] | None = None) -> int:
    global _SHOULD_STOP
    args = parse_args(argv)
    config = load_config(
        args.config,
        robot_sn=args.robot_sn,
        policy_path=args.policy_path,
        residual_policy_path=args.residual_policy_path,
        mode=args.mode,
        residual_lambda=args.residual_lambda,
        device=args.device,
        fps=args.fps,
        dry_run=args.dry_run if args.dry_run else None,
        task=args.task,
        log_dir=args.log_dir,
        gripper_hysteresis=args.gripper_hysteresis,
    )
    if config.inference.policy_path is None:
        print("ERROR: ACT policy path is required.", file=sys.stderr)
        return 2
    if config.inference.mode != "act" and (
        config.inference.residual is None or config.inference.residual.policy_path is None
    ):
        print("ERROR: residual policy path is required for residual modes.", file=sys.stderr)
        return 2
    if config.inference.fps <= 0:
        print(f"ERROR: fps must be positive, got {config.inference.fps}", file=sys.stderr)
        return 2

    os.environ["ROS_DOMAIN_ID"] = str(config.ros.domain_id)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    reader: BWObservationReader | None = None
    executor: SingleThreadedExecutor | None = None
    spin_thread: threading.Thread | None = None
    csv_logger = ActionCSVLogger(config.inference.log_dir, gripper_config=config.inference.gripper)
    step_count = 0
    filter_state = ActionFilterState()
    gripper_controller = BinaryGripperController(config.inference.gripper)

    try:
        print("[1/4] Loading frozen ACT policy...")
        act_bundle = load_policy_bundle(
            config.inference.policy_path,
            device=config.inference.device,
            use_amp=config.inference.use_amp,
        )
        if config.inference.reset_policy_on_start and hasattr(act_bundle.policy, "reset"):
            act_bundle.policy.reset()
        expected_shapes = act_bundle.image_shapes
        print(f"ACT policy dir: {act_bundle.policy_dir}")
        print(f"ACT fingerprint: {act_bundle.fingerprint[:16]}...")
        print(f"Device: {act_bundle.device}, use_amp={act_bundle.use_amp}")
        print(
            "ACT execution: fresh forward every control step; "
            f"temporal_ensemble_coeff={getattr(act_bundle.policy.config, 'temporal_ensemble_coeff', None)}"
        )

        residual_bundle = None
        if config.inference.mode != "act":
            print(f"Loading {config.inference.mode} checkpoint...")
            residual_bundle = load_residual_policy(
                config.inference.residual.policy_path,
                device=act_bundle.device,
            )
            _validate_residual_pair(config.inference.mode, act_bundle, residual_bundle)
            print(f"Residual checkpoint: {residual_bundle.checkpoint_path}")
            print(
                f"Residual type={residual_bundle.policy_type}, obs_dim={residual_bundle.input_dim}, "
                f"visual_dim={residual_bundle.visual_feature_dim}"
            )
            runtime_lambda = (
                residual_bundle.residual_lambda
                if config.inference.residual.lambda_ is None
                else float(config.inference.residual.lambda_)
            )
            if not np.isclose(runtime_lambda, residual_bundle.residual_lambda):
                print(
                    "WARNING: runtime residual lambda differs from training: "
                    f"runtime={runtime_lambda}, training={residual_bundle.residual_lambda}"
                )
            runtime_hysteresis = config.inference.gripper.hysteresis
            training_label_config = residual_bundle.gripper_control
            mismatch = []
            comparisons = (
                ("hysteresis_enabled", runtime_hysteresis.enabled),
                ("open_value", config.inference.gripper.open_value),
                ("close_value", config.inference.gripper.close_value),
                (
                    "residual_confidence_threshold",
                    config.inference.gripper.residual_confidence_threshold,
                ),
                ("residual_confirm_frames", config.inference.gripper.residual_confirm_frames),
                ("min_hold_s", config.inference.gripper.min_hold_s),
                ("open_threshold", runtime_hysteresis.open_threshold),
                ("single_threshold", runtime_hysteresis.single_threshold),
                ("close_threshold", runtime_hysteresis.close_threshold),
            )
            for key, runtime_value in comparisons:
                training_value = training_label_config.get(key)
                if training_value is not None and (
                    bool(training_value) != bool(runtime_value)
                    if key == "hysteresis_enabled"
                    else int(training_value) != int(runtime_value)
                    if key == "residual_confirm_frames"
                    else not np.isclose(float(training_value), float(runtime_value), rtol=0.0, atol=1e-9)
                ):
                    mismatch.append(f"{key}: runtime={runtime_value}, training={training_value}")
            if mismatch:
                print(
                    "WARNING: runtime gripper control differs from residual checkpoint metadata: "
                    + "; ".join(mismatch)
                )

        print("\n[2/4] Connecting to ROS2 topics...")
        init_ros(config)
        reader = BWObservationReader(config)
        executor = SingleThreadedExecutor()
        executor.add_node(reader)
        spin_thread = threading.Thread(target=executor.spin, name="bw_ros_executor", daemon=True)
        spin_thread.start()
        print("\n[3/4] Waiting for complete observation input...")
        if not reader.wait_for_first_messages(timeout_s=config.inference.warmup_timeout_s, spin=False):
            print("ERROR: timed out waiting for required topics:", file=sys.stderr)
            for topic in reader.missing_inputs():
                print(f"  - {topic}", file=sys.stderr)
            return 3

        if not _camera_streams_ready(reader, spin=False):
            return 4

        first_sample = None
        deadline = time.monotonic() + max(config.inference.warmup_timeout_s, 1.0)
        while rclpy.ok() and time.monotonic() < deadline:
            first_sample = reader.get_latest_sample(
                expected_image_shapes=expected_shapes,
                require_new_images=False,
            )
            if first_sample is not None:
                break
            time.sleep(0.01)
        if first_sample is None:
            print("ERROR: no complete valid observation could be assembled.", file=sys.stderr)
            if reader.last_error:
                print(f"Last error: {reader.last_error}", file=sys.stderr)
            return 5

        print("Observation check:")
        print(f"  - observation.state shape={first_sample.observation_state.shape}")
        for camera_name, image in first_sample.images.items():
            source = first_sample.image_sources[camera_name]
            print(
                f"  - observation.images.{camera_name} "
                f"source={source.width}x{source.height} {source.encoding} step={source.step} "
                f"-> ACT RGB shape={image.shape} dtype={image.dtype}"
            )

        print("\n[4/4] Running inference loop. Press Ctrl+C to stop.")
        print(f"Mode: {config.inference.mode}")
        print(f"Publishing arm action to:     {config.robot.output_topics.arm_action}")
        print(f"Publishing gripper action to: {config.robot.output_topics.gripper_action}")
        gripper_config = config.inference.gripper
        print(
            "Gripper control: "
            f"hysteresis={gripper_config.hysteresis.enabled} "
            f"thresholds={gripper_config.hysteresis.open_threshold:g}/"
            f"{gripper_config.hysteresis.single_threshold:g}/"
            f"{gripper_config.hysteresis.close_threshold:g} "
            f"commands={gripper_config.open_value:g}/{gripper_config.close_value:g} "
            f"confidence={gripper_config.residual_confidence_threshold:g} "
            f"confirm_frames={gripper_config.residual_confirm_frames} "
            f"min_hold_s={gripper_config.min_hold_s:g}"
        )
        if config.inference.dry_run:
            print("Dry-run enabled: no output/debug messages are published.")

        target_dt = 1.0 / float(config.inference.fps)
        stream_config = config.inference.camera_stream
        assert stream_config is not None
        next_time = time.monotonic()
        last_log_time = time.monotonic()
        missed_deadlines = 0
        camera_wait_cycles = 0
        zero_arm_delta = np.zeros(len(JOINT_NAMES) - len(GRIPPER_JOINT_INDICES), dtype=np.float32)
        residual_limits = (
            residual_bundle.residual_limits
            if residual_bundle is not None
            else np.full(len(JOINT_NAMES), 0.03, dtype=np.float32)
        )
        runtime_residual_lambda = (
            residual_bundle.residual_lambda
            if residual_bundle is not None and config.inference.residual.lambda_ is None
            else float(config.inference.residual.lambda_ or 0.2)
        )

        while rclpy.ok() and not _SHOULD_STOP:
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(next_time - now, 0.002))
                continue
            sample = reader.get_latest_sample(
                expected_image_shapes=expected_shapes,
                require_new_images=stream_config.require_new_frames,
            )
            if sample is None:
                camera_wait_cycles += 1
                if reader.last_error:
                    reader.get_logger().warning(f"Skipping invalid observation: {reader.last_error}")
                time.sleep(0.001)
                continue
            sample_time = time.monotonic()
            lag = sample_time - next_time
            skipped_deadlines = int(lag / target_dt) if lag >= target_dt else 0
            missed_deadlines += skipped_deadlines
            next_time += (skipped_deadlines + 1) * target_dt

            try:
                observation = sample.to_lerobot_observation()
                if residual_bundle is None:
                    raw_act = infer_action(
                        act_bundle,
                        observation,
                        task=config.inference.task,
                        robot_type=config.inference.robot_type,
                    )
                    act_feature = None
                else:
                    raw_act, act_feature = infer_action_with_shared_visual_feature(
                        act_bundle,
                        observation,
                        task=config.inference.task,
                        robot_type=config.inference.robot_type,
                    )
                action_act = align_action_vector(raw_act, sample.observation_state, expected_dim=len(JOINT_NAMES))
                delta_norm_arm = zero_arm_delta.copy()
                gripper_classes = np.zeros(2, dtype=np.int64)
                gripper_confidences = np.ones(2, dtype=np.float32)
                delta_joint = np.zeros(len(JOINT_NAMES), dtype=np.float32)
                action_final_raw = action_act.copy()
                if residual_bundle is not None:
                    assert act_feature is not None
                    residual_obs = build_residual_runtime_obs(
                        observation_state=sample.observation_state,
                        action_act=action_act,
                        act_feature=act_feature,
                    )
                    residual_output = infer_residual_action(
                        residual_bundle,
                        residual_obs,
                        deterministic=config.inference.residual.deterministic,
                    )
                    delta_norm_arm = residual_output.arm_delta_normalized
                    gripper_classes = residual_output.gripper_classes
                    gripper_confidences = residual_output.gripper_confidences
                    delta_joint, action_final_raw = compose_residual_action(
                        action_act,
                        delta_norm_arm,
                        residual_limits=residual_limits,
                        residual_lambda=runtime_residual_lambda,
                        delta_is_normalized=True,
                    )

                action_final = apply_clamp(action_final_raw, config.inference.clamp)
                action_final = apply_smoothing(
                    action_final,
                    sample.observation_state,
                    config.inference.smoothing,
                    filter_state,
                )
                gripper_result = gripper_controller.step(
                    action_act[GRIPPER_JOINT_INDICES],
                    gripper_classes,
                    gripper_confidences,
                    now_s=sample_time,
                )
                action_composed_debug = action_final_raw.copy()
                action_composed_debug[GRIPPER_JOINT_INDICES] = gripper_result.candidate_action
                action_final[GRIPPER_JOINT_INDICES] = gripper_result.final_action
                filter_state.previous_action = action_final.astype(np.float32, copy=True)
                stamp = reader.get_clock().now().to_msg()
                arm_msg, gripper_msg = split_action_to_joint_states(
                    action_final,
                    stamp=stamp,
                    gripper_name_style=config.inference.gripper_name_style,
                    arm_velocity_limit=config.inference.arm_velocity_limit,
                )
                debug_act = vector_to_joint_state(action_act, stamp=stamp)
                debug_delta = vector_to_joint_state(delta_joint, stamp=stamp)
                debug_composed = vector_to_joint_state(action_composed_debug, stamp=stamp)
                debug_final = vector_to_joint_state(action_final, stamp=stamp)
                reader.publish_action(arm_msg, gripper_msg, dry_run=config.inference.dry_run)
                reader.publish_debug_actions(
                    debug_act,
                    debug_delta,
                    debug_composed,
                    debug_final,
                    gripper_result.raw_classes,
                    dry_run=config.inference.dry_run,
                )
                csv_logger.write(
                    step=step_count,
                    control_source=sample.control_source,
                    action_act=action_act,
                    delta=delta_joint,
                    action_final=action_final,
                    gripper_classes=gripper_result.raw_classes,
                    gripper_confidences=gripper_result.confidences,
                    gripper_hysteresis_enabled=config.inference.gripper.hysteresis.enabled,
                )
                step_count += 1

                if config.inference.log_every_n_steps > 0 and step_count % config.inference.log_every_n_steps == 0:
                    elapsed = max(time.monotonic() - last_log_time, 1e-6)
                    recent_hz = config.inference.log_every_n_steps / elapsed
                    last_log_time = time.monotonic()
                    log_message = (
                        f"step={step_count} recent_hz={recent_hz:.1f} mode={config.inference.mode} "
                        f"missed_deadlines={missed_deadlines} camera_wait_cycles={camera_wait_cycles} "
                        f"delta_abs_max={float(np.max(np.abs(delta_joint))):.5f} "
                        f"left_gripper={float(action_final[JOINT_NAMES.index('left_gripper_joint')]):.6f} "
                        f"right_gripper={float(action_final[JOINT_NAMES.index('right_gripper_joint')]):.6f}"
                    )
                    if recent_hz < stream_config.minimum_fps:
                        reader.get_logger().warning(
                            f"Inference is below the 30 FPS contract: {log_message}"
                        )
                    else:
                        reader.get_logger().info(log_message)
                    missed_deadlines = 0
                    camera_wait_cycles = 0
            except Exception as exc:
                import traceback

                reader.get_logger().error(f"Inference/publish step failed: {exc}\n{traceback.format_exc()}")
                time.sleep(0.05)

            if args.max_steps > 0 and step_count >= args.max_steps:
                print(f"Reached --max-steps {args.max_steps}; exiting.")
                break

        print(f"Stopped after {step_count} inference step(s).")
        return 0
    finally:
        csv_logger.close()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        if reader is not None:
            reader.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
