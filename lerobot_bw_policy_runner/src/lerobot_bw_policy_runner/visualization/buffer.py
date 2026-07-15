"""Thread-safe synchronization and bounded history for action debug streams."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time
from typing import Iterable, Literal

import numpy as np

from ..constants import JOINT_NAMES

ActionStream = Literal["act", "delta", "composed", "final"]
ACTION_STREAMS: tuple[ActionStream, ...] = ("act", "delta", "composed", "final")


@dataclass(frozen=True, slots=True)
class ActionSample:
    timestamp_ns: int
    act: np.ndarray
    delta: np.ndarray
    composed: np.ndarray
    final: np.ndarray


@dataclass(frozen=True, slots=True)
class ActionHistorySnapshot:
    timestamps_ns: np.ndarray
    act: np.ndarray
    delta: np.ndarray
    composed: np.ndarray
    final: np.ndarray
    last_received_monotonic: float | None

    @property
    def sample_count(self) -> int:
        return int(self.timestamps_ns.size)


def joint_values_by_name(names: Iterable[str], positions: Iterable[float]) -> np.ndarray:
    """Return positions in canonical JOINT_NAMES order."""
    name_list = [str(name) for name in names]
    values = np.asarray(list(positions), dtype=np.float32).reshape(-1)
    if len(name_list) != values.size:
        raise ValueError(f"JointState name/position length mismatch: {len(name_list)}/{values.size}")
    if len(set(name_list)) != len(name_list):
        raise ValueError("JointState contains duplicate joint names")
    value_by_name = dict(zip(name_list, values))
    missing = [name for name in JOINT_NAMES if name not in value_by_name]
    if missing:
        raise ValueError(f"JointState is missing required joints: {', '.join(missing)}")
    ordered = np.asarray([value_by_name[name] for name in JOINT_NAMES], dtype=np.float32)
    if not np.all(np.isfinite(ordered)):
        raise ValueError("JointState contains NaN or Inf")
    return ordered


class ActionHistory:
    """Combine four timestamped streams and retain a bounded time window."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        *,
        max_samples: int | None = None,
        pending_limit: int = 64,
        clock_reset_tolerance_seconds: float = 0.5,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if pending_limit < 4:
            raise ValueError("pending_limit must be at least 4")
        self.window_seconds = float(window_seconds)
        self._window_ns = int(self.window_seconds * 1_000_000_000)
        self._max_samples = int(max_samples or max(256, math.ceil(self.window_seconds * 120.0)))
        if self._max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._pending_limit = int(pending_limit)
        self._clock_reset_tolerance_ns = int(clock_reset_tolerance_seconds * 1_000_000_000)
        self._lock = threading.Lock()
        self._history: deque[ActionSample] = deque(maxlen=self._max_samples)
        self._pending: dict[int, dict[ActionStream, np.ndarray]] = {}
        self._pending_last_seen: dict[int, float] = {}
        self._last_timestamp_ns: int | None = None
        self._last_received_monotonic: float | None = None

    def add_message(
        self,
        stream: ActionStream,
        timestamp_ns: int,
        names: Iterable[str],
        positions: Iterable[float],
        *,
        received_monotonic: float | None = None,
    ) -> bool:
        """Add one stream message; return True when a complete sample is committed."""
        if stream not in ACTION_STREAMS:
            raise ValueError(f"Unknown action stream: {stream}")
        stamp = int(timestamp_ns)
        values = joint_values_by_name(names, positions)
        values.setflags(write=False)
        received = time.monotonic() if received_monotonic is None else float(received_monotonic)

        with self._lock:
            pending = self._pending.setdefault(stamp, {})
            pending[stream] = values
            self._pending_last_seen[stamp] = received
            self._prune_pending(received)
            if not all(name in pending for name in ACTION_STREAMS):
                return False

            if self._last_timestamp_ns is not None and stamp < (
                self._last_timestamp_ns - self._clock_reset_tolerance_ns
            ):
                self._history.clear()
                self._pending = {stamp: pending}
                self._pending_last_seen = {stamp: received}
                self._last_timestamp_ns = None
            elif self._last_timestamp_ns is not None and stamp <= self._last_timestamp_ns:
                self._pending.pop(stamp, None)
                self._pending_last_seen.pop(stamp, None)
                return False

            sample = ActionSample(
                timestamp_ns=stamp,
                act=pending["act"],
                delta=pending["delta"],
                composed=pending["composed"],
                final=pending["final"],
            )
            self._history.append(sample)
            self._last_timestamp_ns = stamp
            self._last_received_monotonic = received
            self._pending.pop(stamp, None)
            self._pending_last_seen.pop(stamp, None)
            self._drop_expired_history(stamp)
            for old_stamp in [value for value in self._pending if value <= stamp]:
                self._pending.pop(old_stamp, None)
                self._pending_last_seen.pop(old_stamp, None)
            return True

    def snapshot(self) -> ActionHistorySnapshot:
        with self._lock:
            samples = tuple(self._history)
            last_received = self._last_received_monotonic
        if not samples:
            empty_matrix = np.empty((0, len(JOINT_NAMES)), dtype=np.float32)
            return ActionHistorySnapshot(
                timestamps_ns=np.empty(0, dtype=np.int64),
                act=empty_matrix,
                delta=empty_matrix.copy(),
                composed=empty_matrix.copy(),
                final=empty_matrix.copy(),
                last_received_monotonic=last_received,
            )
        return ActionHistorySnapshot(
            timestamps_ns=np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64),
            act=np.stack([sample.act for sample in samples]),
            delta=np.stack([sample.delta for sample in samples]),
            composed=np.stack([sample.composed for sample in samples]),
            final=np.stack([sample.final for sample in samples]),
            last_received_monotonic=last_received,
        )

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._history)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._pending.clear()
            self._pending_last_seen.clear()
            self._last_timestamp_ns = None
            self._last_received_monotonic = None

    def _drop_expired_history(self, newest_stamp: int) -> None:
        cutoff = newest_stamp - self._window_ns
        while self._history and self._history[0].timestamp_ns < cutoff:
            self._history.popleft()

    def _prune_pending(self, now: float) -> None:
        stale = [stamp for stamp, last_seen in self._pending_last_seen.items() if now - last_seen > 1.0]
        for stamp in stale:
            self._pending.pop(stamp, None)
            self._pending_last_seen.pop(stamp, None)
        if len(self._pending) <= self._pending_limit:
            return
        by_age = sorted(self._pending_last_seen, key=self._pending_last_seen.get)
        for stamp in by_age[: len(self._pending) - self._pending_limit]:
            self._pending.pop(stamp, None)
            self._pending_last_seen.pop(stamp, None)
