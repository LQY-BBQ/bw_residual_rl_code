"""Unique ROS camera-frame tracking and rate statistics."""
from __future__ import annotations

from dataclasses import dataclass
import math


def image_stamp_ns(msg: object) -> int | None:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    seconds = int(getattr(stamp, "sec", 0))
    nanoseconds = int(getattr(stamp, "nanosec", 0))
    value = seconds * 1_000_000_000 + nanoseconds
    return value if value != 0 else None


@dataclass(slots=True)
class CameraStreamStatus:
    unique_frames: int
    duplicate_frames: int
    unstamped_frames: int
    duration_s: float
    age_s: float

    @property
    def fps(self) -> float:
        return self.unique_frames / self.duration_s if self.duration_s > 0 else 0.0


@dataclass(slots=True)
class _CameraState:
    sequence: int = 0
    callbacks: int = 0
    duplicates: int = 0
    unstamped: int = 0
    last_stamp_ns: int | None = None
    received_monotonic: float = float("-inf")


class CameraStreamTracker:
    def __init__(self, camera_names: list[str]) -> None:
        self._states = {name: _CameraState() for name in camera_names}

    def update(self, camera_name: str, msg: object, *, received_monotonic: float) -> bool:
        state = self._states[camera_name]
        state.callbacks += 1
        stamp_ns = image_stamp_ns(msg)
        if stamp_ns is None:
            state.unstamped += 1
        if stamp_ns is not None and stamp_ns == state.last_stamp_ns:
            state.duplicates += 1
            return False
        state.sequence += 1
        state.last_stamp_ns = stamp_ns
        state.received_monotonic = float(received_monotonic)
        return True

    def sequences(self) -> dict[str, int]:
        return {name: state.sequence for name, state in self._states.items()}

    def received_times(self) -> dict[str, float]:
        return {name: state.received_monotonic for name, state in self._states.items()}

    def counters(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (state.sequence, state.duplicates, state.unstamped)
            for name, state in self._states.items()
        }

    def statuses(
        self,
        start: dict[str, tuple[int, int, int]],
        *,
        duration_s: float,
        now_monotonic: float,
    ) -> dict[str, CameraStreamStatus]:
        statuses: dict[str, CameraStreamStatus] = {}
        for name, state in self._states.items():
            start_unique, start_duplicates, start_unstamped = start[name]
            age = now_monotonic - state.received_monotonic
            statuses[name] = CameraStreamStatus(
                unique_frames=max(state.sequence - start_unique, 0),
                duplicate_frames=max(state.duplicates - start_duplicates, 0),
                unstamped_frames=max(state.unstamped - start_unstamped, 0),
                duration_s=max(float(duration_s), 0.0),
                age_s=age if math.isfinite(age) else float("inf"),
            )
        return statuses
