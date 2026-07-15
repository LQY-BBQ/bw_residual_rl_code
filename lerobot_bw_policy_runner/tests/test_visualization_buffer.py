from __future__ import annotations

import numpy as np
import pytest

from lerobot_bw_policy_runner.constants import JOINT_NAMES
from lerobot_bw_policy_runner.visualization.buffer import ACTION_STREAMS, ActionHistory, joint_values_by_name


def _add_complete_sample(
    history: ActionHistory,
    timestamp_ns: int,
    *,
    base: float = 0.0,
    received_monotonic: float = 1.0,
) -> None:
    canonical = np.arange(len(JOINT_NAMES), dtype=np.float32) + base
    names = list(reversed(JOINT_NAMES))
    positions = list(reversed(canonical.tolist()))
    for stream in ("final", "act", "composed", "delta"):
        history.add_message(
            stream,
            timestamp_ns,
            names,
            positions,
            received_monotonic=received_monotonic,
        )


def test_combines_out_of_order_streams_and_reorders_joints() -> None:
    history = ActionHistory(window_seconds=10.0)
    _add_complete_sample(history, 2_000_000_000, base=3.0)

    snapshot = history.snapshot()
    assert snapshot.sample_count == 1
    np.testing.assert_array_equal(
        snapshot.act[0],
        np.arange(len(JOINT_NAMES), dtype=np.float32) + 3.0,
    )
    assert snapshot.timestamps_ns.tolist() == [2_000_000_000]


def test_never_combines_messages_from_different_timestamps() -> None:
    history = ActionHistory(window_seconds=10.0)
    values = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    for index, stream in enumerate(ACTION_STREAMS):
        history.add_message(stream, index + 1, JOINT_NAMES, values, received_monotonic=1.0)
    assert history.sample_count == 0


def test_history_is_time_bounded_and_sample_bounded() -> None:
    history = ActionHistory(window_seconds=1.0, max_samples=3)
    for index, timestamp_ns in enumerate((0, 500_000_000, 1_000_000_000, 1_500_000_000)):
        _add_complete_sample(history, timestamp_ns, base=float(index), received_monotonic=float(index))

    snapshot = history.snapshot()
    assert snapshot.sample_count == 3
    assert snapshot.timestamps_ns.tolist() == [500_000_000, 1_000_000_000, 1_500_000_000]


def test_joint_validation_rejects_missing_duplicate_and_non_finite_values() -> None:
    values = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    with pytest.raises(ValueError, match="missing required joints"):
        joint_values_by_name(JOINT_NAMES[:-1], values[:-1])
    duplicate_names = list(JOINT_NAMES)
    duplicate_names[-1] = duplicate_names[0]
    with pytest.raises(ValueError, match="duplicate"):
        joint_values_by_name(duplicate_names, values)
    values[-1] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        joint_values_by_name(JOINT_NAMES, values)
