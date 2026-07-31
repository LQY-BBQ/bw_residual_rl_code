from __future__ import annotations

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from lerobot_bw_policy_runner.action_utils import (
    ActionFilterState,
    apply_smoothing,
    compose_residual_action,
    split_action_to_joint_states,
)
from lerobot_bw_policy_runner.config import ActionSmoothingConfig, default_config_path, load_config
from lerobot_bw_policy_runner.constants import ARM_JOINT_INDICES, ARM_JOINT_NAMES, GRIPPER_JOINT_INDICES, JOINT_NAMES


def test_composed_and_final_debug_action_semantics() -> None:
    act = np.linspace(-0.4, 0.4, len(JOINT_NAMES), dtype=np.float32)
    normalized_delta = np.linspace(-1.0, 1.0, len(ARM_JOINT_NAMES), dtype=np.float32)
    limits = np.full(len(ARM_JOINT_NAMES), 0.1, dtype=np.float32)
    delta, composed = compose_residual_action(
        act,
        normalized_delta,
        residual_limits=limits,
        residual_lambda=0.25,
    )
    np.testing.assert_allclose(composed[ARM_JOINT_INDICES], act[ARM_JOINT_INDICES] + 0.25 * delta[ARM_JOINT_INDICES])
    np.testing.assert_array_equal(delta[GRIPPER_JOINT_INDICES], 0.0)
    np.testing.assert_array_equal(composed[GRIPPER_JOINT_INDICES], act[GRIPPER_JOINT_INDICES])

    final = apply_smoothing(
        composed,
        np.zeros(len(JOINT_NAMES), dtype=np.float32),
        ActionSmoothingConfig(enabled=True, alpha=0.5),
        ActionFilterState(),
    )
    arm_msg, gripper_msg = split_action_to_joint_states(final, stamp=Time(sec=1, nanosec=2))
    value_by_name = dict(zip(arm_msg.name, arm_msg.position))
    value_by_name.update(zip(gripper_msg.name, gripper_msg.position))
    np.testing.assert_allclose(
        [value_by_name[name] for name in ARM_JOINT_NAMES],
        [final[JOINT_NAMES.index(name)] for name in ARM_JOINT_NAMES],
    )
    assert value_by_name["left_gripper_joint"] == float(final[JOINT_NAMES.index("left_gripper_joint")])
    assert value_by_name["right_gripper_joint"] == float(final[JOINT_NAMES.index("right_gripper_joint")])


def test_default_composed_topic_expands_robot_serial_number() -> None:
    config = load_config(default_config_path(), robot_sn="BW_TEST123")
    assert config.robot.output_topics.debug_action_composed == "/BW_TEST123/Policy/debug/action_composed"


def test_continuous_16d_residual_is_rejected() -> None:
    with pytest.raises(ValueError, match="must have 14 arm values"):
        compose_residual_action(
            np.zeros(16, dtype=np.float32),
            np.zeros(16, dtype=np.float32),
            residual_limits=np.ones(16, dtype=np.float32),
            residual_lambda=0.2,
        )
