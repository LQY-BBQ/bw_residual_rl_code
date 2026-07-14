"""Non-blocking keyboard marker for RL data collection.

Keys are read from the terminal while the collector is running:
- a: left block has been placed successfully
- d: right block has been placed successfully; also ends the episode as success
- g: mark the episode as success and stop recording
- j: mark the episode as failure and stop recording

This module intentionally does not depend on ROS, so it can be used inside the
normal rclpy spin loop without adding another node or topic.
"""
from __future__ import annotations

from dataclasses import dataclass
import select
import sys
import termios
import tty


@dataclass(slots=True)
class MarkerDecision:
    """Reward/done decision for one frame."""

    reward: float = 0.0
    done: bool = False
    success: bool = False
    stop_reason: str | None = None


class KeyboardRewardMarker:
    """Collect sparse RL reward labels from terminal key presses.

    The marker is edge-triggered and stage keys are one-shot:
    pressing ``a`` repeatedly only gives the left-stage reward once;
    pressing ``d`` repeatedly only gives the right-stage reward once.
    """

    LEFT_KEY = "a"
    RIGHT_KEY = "d"
    SUCCESS_KEY = "g"
    FAILURE_KEY = "j"

    def __init__(self, *, enable: bool = True) -> None:
        self.enabled = bool(enable) and sys.stdin.isatty()
        self.left_done = False
        self.right_done = False
        self.left_done_frame: int | None = None
        self.right_done_frame: int | None = None
        self._old_termios: list[int | bytes] | None = None

    def __enter__(self) -> "KeyboardRewardMarker":
        if self.enabled:
            self._old_termios = termios.tcgetattr(sys.stdin.fileno())
            # cbreak keeps Ctrl+C as SIGINT but allows single-character reads.
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        del exc_type, exc, tb
        if self.enabled and self._old_termios is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def help_text(self) -> str:
        if not self.enabled:
            return "Keyboard reward marker disabled because stdin is not a TTY."
        return (
            "RL keyboard labels: "
            "a=left block done(+1), "
            "d=right block done(+1) and success stop, "
            "g=success stop(+1), "
            "j=failure stop"
        )

    def _read_keys(self) -> list[str]:
        if not self.enabled:
            return []
        keys: list[str] = []
        # Drain all currently buffered keys. This avoids missing a quick press
        # while the collector is sleeping between frames.
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch.lower())
        return keys

    def poll(self, *, frame_index: int) -> MarkerDecision:
        """Return the reward/done labels that should be applied to frame_index."""
        decision = MarkerDecision()
        for key in self._read_keys():
            if key == self.LEFT_KEY:
                if not self.left_done:
                    self.left_done = True
                    self.left_done_frame = int(frame_index)
                    decision.reward += 1.0
                    print(f"\n[MARK] frame={frame_index}: left block done, reward += 1")
                else:
                    print(f"\n[MARK] frame={frame_index}: ignored duplicate left-block key 'a'")

            elif key == self.RIGHT_KEY:
                if not self.right_done:
                    self.right_done = True
                    self.right_done_frame = int(frame_index)
                    decision.reward += 1.0
                    print(f"\n[MARK] frame={frame_index}: right block done, reward += 1")
                else:
                    print(f"\n[MARK] frame={frame_index}: ignored duplicate right-block key 'd'")
                # User selected option B: right block done means episode success.
                decision.reward += 1.0
                decision.done = True
                decision.success = True
                decision.stop_reason = "right_block_done_success"
                print(f"[MARK] frame={frame_index}: success stop, reward += 1")

            elif key == self.SUCCESS_KEY:
                decision.reward += 1.0
                decision.done = True
                decision.success = True
                decision.stop_reason = "manual_success"
                print(f"\n[MARK] frame={frame_index}: manual success stop, reward += 1")

            elif key == self.FAILURE_KEY:
                decision.done = True
                decision.success = False
                decision.stop_reason = "manual_failure"
                print(f"\n[MARK] frame={frame_index}: manual failure stop")

        return decision
