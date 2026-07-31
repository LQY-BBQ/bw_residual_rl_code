from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from lerobot_bw_data_collector.dataset_writer import build_features, build_frame


def test_build_features_preserves_each_camera_resolution() -> None:
    images = {
        "env_cam": np.zeros((480, 640, 3), dtype=np.uint8),
        "left_wrist_cam": np.zeros((270, 480, 3), dtype=np.uint8),
        "right_wrist_cam": np.zeros((270, 480, 3), dtype=np.uint8),
    }

    features = build_features(images, use_videos=True)

    assert features["observation.images.env_cam"]["shape"] == (480, 640, 3)
    assert features["observation.images.left_wrist_cam"]["shape"] == (270, 480, 3)
    assert features["observation.images.right_wrist_cam"]["shape"] == (270, 480, 3)


def _rl_sample() -> SimpleNamespace:
    vector = np.zeros(16, dtype=np.float32)
    vector[[7, 15]] = [0.0, 0.8]
    return SimpleNamespace(
        observation_state=np.zeros(16, dtype=np.float32),
        action=vector.copy(),
        images={},
        control_source=1,
        is_intervention=False,
        has_human_action=False,
        action_act=np.linspace(0.0, 0.8, 16, dtype=np.float32),
        action_rl_delta=np.zeros(16, dtype=np.float32),
        action_human=np.zeros(16, dtype=np.float32),
        action_executed=vector.copy(),
        gripper_policy_class=np.asarray([1, 2], dtype=np.int64),
        timing={},
    )


def test_rl_schema_keeps_16d_actions_and_adds_gripper_classes() -> None:
    features = build_features({}, use_videos=False, mode="rl")
    assert features["action.gripper_policy_class"] == {
        "dtype": "int64",
        "shape": (2,),
        "names": ["left", "right"],
    }
    frame = build_frame(_rl_sample(), "test", mode="rl")
    for key in ("observation.state", "action", "action.act", "action.rl_delta", "action.human", "action.executed"):
        assert frame[key].shape == (16,)
    np.testing.assert_array_equal(frame["action.rl_delta"][[7, 15]], 0.0)
    np.testing.assert_array_equal(frame["action.gripper_policy_class"], [1, 2])


def test_rl_frame_rejects_continuous_final_gripper_and_nonzero_gripper_delta() -> None:
    sample = _rl_sample()
    sample.action_executed[7] = 0.3
    with pytest.raises(ValueError, match="must be 0.0 or 0.8"):
        build_frame(sample, "test", mode="rl")
    sample = _rl_sample()
    sample.action_rl_delta[15] = 0.01
    with pytest.raises(ValueError, match="gripper entries must be zero"):
        build_frame(sample, "test", mode="rl")
