from __future__ import annotations

from types import SimpleNamespace

from lerobot_bw_policy_runner.camera_stream import CameraStreamTracker


def _message(stamp_ns: int) -> SimpleNamespace:
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)
        )
    )


def test_inference_tracker_only_advances_for_unique_frames() -> None:
    tracker = CameraStreamTracker(["env_cam", "left_wrist_cam", "right_wrist_cam"])
    for camera_name in ("env_cam", "left_wrist_cam", "right_wrist_cam"):
        assert tracker.update(camera_name, _message(100), received_monotonic=1.0)
        assert not tracker.update(camera_name, _message(100), received_monotonic=1.01)
        assert tracker.update(camera_name, _message(200), received_monotonic=1.02)

    assert tracker.sequences() == {
        "env_cam": 2,
        "left_wrist_cam": 2,
        "right_wrist_cam": 2,
    }
