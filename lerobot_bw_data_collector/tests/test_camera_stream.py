from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from lerobot_bw_data_collector.camera_stream import CameraStreamTracker
from lerobot_bw_data_collector.config import default_config_path, load_config


def _message(stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        )
    )


def test_tracker_rejects_duplicate_ros_timestamps() -> None:
    tracker = CameraStreamTracker(["env_cam"])
    start = tracker.counters()

    assert tracker.update("env_cam", _message(1), received_monotonic=10.0)
    assert not tracker.update("env_cam", _message(1), received_monotonic=10.01)
    assert tracker.update("env_cam", _message(2), received_monotonic=10.05)

    status = tracker.statuses(start, duration_s=0.1, now_monotonic=10.1)["env_cam"]
    assert status.unique_frames == 2
    assert status.duplicate_frames == 1
    assert status.fps == pytest.approx(20.0)
    assert status.age_s == pytest.approx(0.05)


def test_default_config_matches_third_generation_camera_contract() -> None:
    config = load_config(default_config_path(), robot_sn="BW_TEST123")

    assert config.dataset.fps == 30
    assert config.cameras.expected_fps == 30.0
    assert config.cameras.require_new_frames
    env = config.cameras.sources["env_cam"]
    assert config.cameras.topics["env_cam"] == "/camera/env_d435/color/image_raw"
    assert (env.width, env.height, env.encoding) == (640, 480, "rgb8")


def test_unstamped_frames_are_visible_to_preflight() -> None:
    tracker = CameraStreamTracker(["env_cam"])
    start = tracker.counters()

    assert tracker.update("env_cam", _message(0), received_monotonic=2.0)

    status = tracker.statuses(start, duration_s=1.0, now_monotonic=2.1)["env_cam"]
    assert status.unstamped_frames == 1


def test_collector_rejects_non_30_fps_override() -> None:
    with pytest.raises(ValueError, match="must match cameras.expected_fps"):
        load_config(default_config_path(), robot_sn="BW_TEST123", fps=15)


def test_collector_rejects_previous_hdmi_contract(tmp_path) -> None:
    raw = yaml.safe_load(default_config_path().read_text(encoding="utf-8"))
    raw["cameras"]["topics"]["env_cam"] = "/env_camera/image_raw"
    raw["cameras"]["sources"]["env_cam"] = {
        "width": 480,
        "height": 270,
        "encoding": "bgr8",
    }
    config_path = tmp_path / "old_camera.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="third-generation BW camera contract"):
        load_config(config_path, robot_sn="BW_TEST123")
