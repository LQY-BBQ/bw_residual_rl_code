from __future__ import annotations

import numpy as np

from lerobot_bw_data_collector.dataset_writer import build_features


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
