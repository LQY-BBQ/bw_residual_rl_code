from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from lerobot_bw_policy_runner.config import (
    GripperControlConfig,
    GripperHysteresisConfig,
    default_config_path,
    load_config,
)
from lerobot_bw_policy_runner.gripper_control import BinaryGripperController
from lerobot_bw_policy_runner.infer_node import parse_args


def _config(
    *,
    hysteresis: bool = True,
    act_confirm_frames: int = 3,
    residual_confirm_frames: int = 3,
    min_hold_s: float = 0.3,
):
    return GripperControlConfig(
        act_confirm_frames=act_confirm_frames,
        residual_confirm_frames=residual_confirm_frames,
        min_hold_s=min_hold_s,
        hysteresis=GripperHysteresisConfig(enabled=hysteresis),
    )


def test_hysteresis_and_single_threshold_are_independent_per_side() -> None:
    controller = BinaryGripperController(_config(hysteresis=True, act_confirm_frames=1, min_hold_s=0.0))
    first = controller.step([0.1, 0.7], [0, 0], [1.0, 1.0], now_s=0.0)
    np.testing.assert_allclose(first.final_action, [0.0, 0.8])
    middle = controller.step([0.3, 0.6], [0, 0], [1.0, 1.0], now_s=0.1)
    np.testing.assert_allclose(middle.final_action, [0.0, 0.8])
    switched = controller.step([0.4, 0.5], [0, 0], [1.0, 1.0], now_s=0.2)
    np.testing.assert_allclose(switched.final_action, [0.8, 0.0])

    no_hysteresis = BinaryGripperController(_config(hysteresis=False, act_confirm_frames=1, min_hold_s=0.0))
    below = no_hysteresis.step([0.449, 0.45], [0, 0], [1.0, 1.0], now_s=0.0)
    np.testing.assert_allclose(below.final_action, [0.0, 0.8])


def test_act_base_transition_requires_consecutive_frames() -> None:
    controller = BinaryGripperController(_config(act_confirm_frames=3, min_hold_s=0.0))
    initial = controller.step([0.8, 0.0], [0, 0], [1.0, 1.0], now_s=0.0)
    np.testing.assert_allclose(initial.final_action, [0.8, 0.0])

    for now_s in (0.01, 0.02):
        pending_open = controller.step([0.49, 0.0], [0, 0], [1.0, 1.0], now_s=now_s)
        np.testing.assert_allclose(pending_open.final_action, [0.8, 0.0])
    opened = controller.step([0.49, 0.0], [0, 0], [1.0, 1.0], now_s=0.03)
    np.testing.assert_allclose(opened.final_action, [0.0, 0.0])

    noisy_close = controller.step([0.41, 0.0], [0, 0], [1.0, 1.0], now_s=0.04)
    np.testing.assert_allclose(noisy_close.final_action, [0.0, 0.0])
    reset = controller.step([0.39, 0.0], [0, 0], [1.0, 1.0], now_s=0.05)
    np.testing.assert_allclose(reset.final_action, [0.0, 0.0])
    for now_s in (0.06, 0.07):
        pending_close = controller.step([0.41, 0.0], [0, 0], [1.0, 1.0], now_s=now_s)
        np.testing.assert_allclose(pending_close.final_action, [0.0, 0.0])
    closed = controller.step([0.41, 0.0], [0, 0], [1.0, 1.0], now_s=0.08)
    np.testing.assert_allclose(closed.final_action, [0.8, 0.0])


def test_residual_confirmation_confidence_keep_base_and_minimum_hold() -> None:
    controller = BinaryGripperController(_config(hysteresis=False, act_confirm_frames=1))
    for frame, now_s in enumerate((0.0, 0.01), start=1):
        result = controller.step([0.0, 0.8], [2, 1], [0.9, 0.9], now_s=now_s)
        np.testing.assert_allclose(result.final_action, [0.0, 0.8])
    confirmed = controller.step([0.0, 0.8], [2, 1], [0.9, 0.9], now_s=0.02)
    np.testing.assert_allclose(confirmed.final_action, [0.8, 0.0])

    # A low-confidence reverse prediction becomes KEEP_BASE, but needs three
    # frames and still cannot reverse the final state during min_hold_s.
    controller.step([0.0, 0.8], [1, 2], [0.6, 0.6], now_s=0.03)
    controller.step([0.0, 0.8], [1, 2], [0.6, 0.6], now_s=0.04)
    held = controller.step([0.0, 0.8], [1, 2], [0.6, 0.6], now_s=0.05)
    np.testing.assert_allclose(held.final_action, [0.8, 0.0])
    released = controller.step([0.0, 0.8], [0, 0], [1.0, 1.0], now_s=0.33)
    np.testing.assert_allclose(released.final_action, [0.0, 0.8])


def test_reset_seeds_human_gripper_state_and_hold_time() -> None:
    controller = BinaryGripperController(_config(act_confirm_frames=1, min_hold_s=0.3))
    controller.reset(np.asarray([0.0, 0.8], dtype=np.float32), now_s=1.0)
    held = controller.step([0.8, 0.0], [0, 0], [1.0, 1.0], now_s=1.1)
    np.testing.assert_allclose(held.final_action, [0.0, 0.8])
    released = controller.step([0.8, 0.0], [0, 0], [1.0, 1.0], now_s=1.31)
    np.testing.assert_allclose(released.final_action, [0.8, 0.0])


def test_cli_hysteresis_flags_are_mutually_exclusive() -> None:
    assert parse_args(["--gripper-hysteresis"]).gripper_hysteresis is True
    assert parse_args(["--no-gripper-hysteresis"]).gripper_hysteresis is False
    assert parse_args([]).gripper_hysteresis is None
    with pytest.raises(SystemExit):
        parse_args(["--gripper-hysteresis", "--no-gripper-hysteresis"])


def test_yaml_default_and_cli_override(tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["robot"]["robot_sn"] = "BW_TEST"
    raw["inference"]["gripper"]["hysteresis"]["enabled"] = False
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.inference.gripper.hysteresis.enabled is False
    assert loaded.inference.gripper.act_confirm_frames == 3
    assert loaded.inference.gripper.hysteresis.open_threshold == pytest.approx(0.50)
    assert loaded.inference.gripper.hysteresis.single_threshold == pytest.approx(0.45)
    assert loaded.inference.gripper.hysteresis.close_threshold == pytest.approx(0.40)
    assert load_config(path, gripper_hysteresis=True).inference.gripper.hysteresis.enabled is True


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["robot"]["robot_sn"] = "BW_TEST"
    raw["inference"]["gripper"]["hysteresis"]["single_threshold"] = 0.9
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds must each be within"):
        load_config(path)
