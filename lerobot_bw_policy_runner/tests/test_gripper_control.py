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


def _config(*, hysteresis: bool = True, confirm_frames: int = 3, min_hold_s: float = 0.3):
    return GripperControlConfig(
        residual_confirm_frames=confirm_frames,
        min_hold_s=min_hold_s,
        hysteresis=GripperHysteresisConfig(enabled=hysteresis),
    )


def test_hysteresis_and_single_threshold_are_independent_per_side() -> None:
    controller = BinaryGripperController(_config(hysteresis=True, confirm_frames=1, min_hold_s=0.0))
    first = controller.step([0.1, 0.5], [0, 0], [1.0, 1.0], now_s=0.0)
    np.testing.assert_allclose(first.final_action, [0.0, 0.8])
    middle = controller.step([0.3, 0.3], [0, 0], [1.0, 1.0], now_s=0.1)
    np.testing.assert_allclose(middle.final_action, [0.0, 0.8])
    switched = controller.step([0.4, 0.2], [0, 0], [1.0, 1.0], now_s=0.2)
    np.testing.assert_allclose(switched.final_action, [0.8, 0.0])

    no_hysteresis = BinaryGripperController(_config(hysteresis=False, confirm_frames=1, min_hold_s=0.0))
    below = no_hysteresis.step([0.299, 0.3], [0, 0], [1.0, 1.0], now_s=0.0)
    np.testing.assert_allclose(below.final_action, [0.0, 0.8])


def test_residual_confirmation_confidence_keep_base_and_minimum_hold() -> None:
    controller = BinaryGripperController(_config(hysteresis=False))
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
    assert load_config(path).inference.gripper.hysteresis.enabled is False
    assert load_config(path, gripper_hysteresis=True).inference.gripper.hysteresis.enabled is True


def test_invalid_threshold_order_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["robot"]["robot_sn"] = "BW_TEST"
    raw["inference"]["gripper"]["hysteresis"]["single_threshold"] = 0.5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="open_threshold < single_threshold < close_threshold"):
        load_config(path)
