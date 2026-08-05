from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_bw_policy_runner.config import HandoverConfig, default_config_path, load_config
from lerobot_bw_policy_runner.constants import ARM_JOINT_INDICES, GRIPPER_JOINT_INDICES, JOINT_NAMES
from lerobot_bw_policy_runner.handover_control import HandoverPhase, PolicyHandoverController


def _action(arm_value: float, grippers: tuple[float, float] = (0.0, 0.8)) -> np.ndarray:
    action = np.full(len(JOINT_NAMES), arm_value, dtype=np.float32)
    action[GRIPPER_JOINT_INDICES] = grippers
    return action


def _controller() -> PolicyHandoverController:
    return PolicyHandoverController(
        HandoverConfig(),
        fps=30.0,
        gripper_hold_s=0.30,
    )


def test_remote_shadow_uses_feedback_despite_large_policy_error() -> None:
    controller = _controller()
    state = _action(0.0, (0.0, 0.0))
    candidate = _action(0.25, (0.8, 0.8))
    teleop_gripper = np.asarray([0.0, 0.8], dtype=np.float32)

    assert controller.observe_control_source(0, state, teleop_gripper) is False
    result = controller.apply(candidate, state, teleop_gripper)

    assert result.phase == HandoverPhase.REMOTE_SHADOW
    assert result.publish_control is True
    np.testing.assert_array_equal(result.command[ARM_JOINT_INDICES], state[ARM_JOINT_INDICES])
    np.testing.assert_array_equal(result.command[GRIPPER_JOINT_INDICES], teleop_gripper)
    assert result.target_error_max == pytest.approx(0.25)
    assert result.command_feedback_error_max == pytest.approx(0.0)


def test_resume_holds_six_frames_then_limits_velocity_and_feedback_lead() -> None:
    controller = _controller()
    state = _action(0.0, (0.0, 0.8))
    candidate = _action(0.25, (0.8, 0.0))
    teleop_gripper = np.asarray([0.0, 0.8], dtype=np.float32)
    controller.observe_control_source(0, state, teleop_gripper)
    assert controller.observe_control_source(1, state, teleop_gripper) is True
    assert controller.observe_control_source(1, state, teleop_gripper) is False

    commands = []
    for _ in range(20):
        result = controller.apply(candidate, state, teleop_gripper)
        assert result.publish_control is True
        commands.append(result.command.copy())

    for command in commands[:6]:
        np.testing.assert_array_equal(command[ARM_JOINT_INDICES], state[ARM_JOINT_INDICES])
    first_resume = commands[6][ARM_JOINT_INDICES]
    np.testing.assert_allclose(first_resume, 0.005, rtol=0.0, atol=1e-7)

    arm_commands = np.stack([command[ARM_JOINT_INDICES] for command in commands])
    per_step = np.abs(np.diff(arm_commands, axis=0))
    assert float(per_step.max()) <= 0.005 + 1e-7
    assert float(np.abs(arm_commands).max()) <= 0.03 + 1e-7
    for command in commands[:9]:
        np.testing.assert_array_equal(command[GRIPPER_JOINT_INDICES], teleop_gripper)


def test_missing_teleop_gripper_blocks_remote_to_inference_handover() -> None:
    controller = _controller()
    state = _action(0.0)
    candidate = _action(0.1)
    controller.observe_control_source(0, state, None)
    assert controller.observe_control_source(1, state, None) is True

    result = controller.apply(candidate, state, None)
    assert result.publish_control is False
    assert result.phase == HandoverPhase.INITIAL_HOLD
    assert "Teleop gripper" in str(result.reason)


def test_starting_in_inference_requires_gripper_only_for_initial_handover() -> None:
    controller = _controller()
    state = _action(0.0)
    candidate = _action(0.1)

    assert controller.observe_control_source(1, state, None) is True
    blocked = controller.apply(candidate, state, None)
    assert blocked.publish_control is False

    teleop_gripper = np.asarray([0.0, 0.8], dtype=np.float32)
    first_hold = controller.apply(candidate, state, teleop_gripper)
    assert first_hold.publish_control is True
    assert controller.observe_control_source(1, state, None) is False


def test_invalid_control_source_blocks_publication() -> None:
    controller = _controller()
    state = _action(0.0)
    candidate = _action(0.1)
    assert controller.observe_control_source(7, state, None) is False
    result = controller.apply(candidate, state, None)
    assert result.publish_control is False
    assert result.phase == HandoverPhase.WAITING_FOR_SOURCE


def test_resume_completes_after_three_close_frames() -> None:
    controller = _controller()
    state = _action(0.0)
    teleop_gripper = np.asarray([0.0, 0.8], dtype=np.float32)
    controller.observe_control_source(0, state, teleop_gripper)
    controller.observe_control_source(1, state, teleop_gripper)
    for _ in range(6):
        controller.apply(_action(0.0), state, teleop_gripper)
    for expected_phase in (
        HandoverPhase.RESUMING,
        HandoverPhase.RESUMING,
        HandoverPhase.INFERENCE,
    ):
        result = controller.apply(_action(0.004), state, teleop_gripper)
        assert result.phase == expected_phase


def test_default_handover_config_and_validation(tmp_path: Path) -> None:
    config = load_config(default_config_path(), robot_sn="BW_TEST")
    assert config.robot.input_topics.teleop_gripper_action == "/BW_TEST/Teleop/gripper_pos"
    assert config.inference.handover.initial_hold_frames == 6
    assert config.inference.handover.resume_max_velocity == pytest.approx(0.15)

    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["robot"]["robot_sn"] = "BW_TEST"
    raw["inference"]["handover"]["resume_max_velocity"] = 0.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="finite and positive"):
        load_config(path)

    raw["inference"]["handover"]["resume_max_velocity"] = 0.15
    raw["robot"]["input_topics"]["teleop_gripper_action"] = None
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).robot.input_topics.teleop_gripper_action is None
