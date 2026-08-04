from unittest.mock import patch

from lerobot_bw_data_collector.keyboard_marker import KeyboardRewardMarker


def poll_keys(marker: KeyboardRewardMarker, keys: list[str], frame_index: int):
    with patch.object(KeyboardRewardMarker, "_read_keys", return_value=keys):
        return marker.poll(frame_index=frame_index)


def test_stacked_success_is_terminal_with_extra_reward() -> None:
    marker = KeyboardRewardMarker(enable=False)

    left = poll_keys(marker, ["a"], frame_index=10)
    stacked = poll_keys(marker, ["s"], frame_index=20)

    assert left.reward == 1.0
    assert left.done is False
    assert left.success is False
    assert stacked.reward == 3.0
    assert stacked.done is True
    assert stacked.success is True
    assert stacked.stop_reason == "stacked_blocks_success"
    assert marker.left_done is True
    assert marker.right_done is True
    assert marker.right_done_frame == 20


def test_help_text_lists_stacked_success_key() -> None:
    marker = KeyboardRewardMarker(enable=False)
    marker.enabled = True

    assert "s=stacked success stop(+3)" in marker.help_text()


def test_terminal_key_ignores_later_buffered_keys() -> None:
    marker = KeyboardRewardMarker(enable=False)

    stacked = poll_keys(marker, ["s", "j"], frame_index=30)

    assert stacked.reward == 3.0
    assert stacked.done is True
    assert stacked.success is True
    assert stacked.stop_reason == "stacked_blocks_success"
