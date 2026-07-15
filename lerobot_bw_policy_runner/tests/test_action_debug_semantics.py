from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time

from lerobot_bw_policy_runner.action_utils import (
    ActionFilterState,
    apply_smoothing,
    compose_residual_action,
    split_action_to_joint_states,
)
from lerobot_bw_policy_runner.config import ActionSmoothingConfig, default_config_path, load_config
from lerobot_bw_policy_runner.constants import ARM_JOINT_NAMES, JOINT_NAMES


def test_composed_and_final_debug_action_semantics() -> None:
    act = np.linspace(-0.4, 0.4, len(JOINT_NAMES), dtype=np.float32)
    normalized_delta = np.linspace(-1.0, 1.0, len(JOINT_NAMES), dtype=np.float32)
    limits = np.full(len(JOINT_NAMES), 0.1, dtype=np.float32)
    delta, composed = compose_residual_action(
        act,
        normalized_delta,
        residual_limits=limits,
        residual_lambda=0.25,
    )
    np.testing.assert_allclose(composed, act + 0.25 * delta)

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
