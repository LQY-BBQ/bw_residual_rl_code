"""Binary gripper control with categorical residual overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .config import GripperControlConfig


class GripperResidualClass(IntEnum):
    KEEP_BASE = 0
    FORCE_OPEN = 1
    FORCE_CLOSE = 2


GRIPPER_CLASS_NAMES = tuple(item.name for item in GripperResidualClass)
_THRESHOLD_EPSILON = 1e-6


@dataclass(slots=True)
class GripperControlResult:
    raw_classes: np.ndarray
    confidences: np.ndarray
    base_closed: np.ndarray
    candidate_action: np.ndarray
    final_action: np.ndarray


@dataclass(slots=True)
class _SideState:
    initialized: bool = False
    base_closed: bool = False
    pending_base_closed: bool = False
    pending_base_count: int = 0
    final_closed: bool = False
    active_class: GripperResidualClass = GripperResidualClass.KEEP_BASE
    pending_class: GripperResidualClass = GripperResidualClass.KEEP_BASE
    pending_count: int = 0
    last_change_s: float = float("-inf")


@dataclass(slots=True)
class BinaryGripperController:
    config: GripperControlConfig
    sides: list[_SideState] = field(default_factory=lambda: [_SideState(), _SideState()])

    def reset(self, action: np.ndarray | None = None, *, now_s: float = 0.0) -> None:
        """Clear policy history, optionally seeding the final state from Teleop."""
        self.sides = [_SideState(), _SideState()]
        if action is None:
            return
        values = np.asarray(action, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(values)):
            raise ValueError("Gripper reset action contains NaN or Inf")
        valid = np.isclose(values, self.config.open_value, rtol=0.0, atol=1e-6) | np.isclose(
            values, self.config.close_value, rtol=0.0, atol=1e-6
        )
        if not np.all(valid):
            raise ValueError(
                "Gripper reset action must use configured open/close endpoints, "
                f"got {values.tolist()}"
            )
        for side, value in zip(self.sides, values):
            closed = bool(value >= (self.config.open_value + self.config.close_value) * 0.5)
            side.initialized = True
            side.base_closed = closed
            side.pending_base_closed = closed
            side.final_closed = closed
            side.last_change_s = float(now_s)

    def _update_base(self, state: _SideState, score: float) -> None:
        hysteresis = self.config.hysteresis
        if not state.initialized:
            threshold = hysteresis.close_threshold if hysteresis.enabled else hysteresis.single_threshold
            state.base_closed = score >= threshold - _THRESHOLD_EPSILON
            state.pending_base_closed = state.base_closed
            state.final_closed = state.base_closed
            state.initialized = True
            return

        if not hysteresis.enabled:
            requested_closed = score >= hysteresis.single_threshold - _THRESHOLD_EPSILON
        elif state.base_closed:
            requested_closed = score > hysteresis.open_threshold + _THRESHOLD_EPSILON
        else:
            requested_closed = score >= hysteresis.close_threshold - _THRESHOLD_EPSILON

        if requested_closed == state.base_closed:
            state.pending_base_closed = state.base_closed
            state.pending_base_count = 0
            return
        if requested_closed != state.pending_base_closed:
            state.pending_base_closed = requested_closed
            state.pending_base_count = 1
        else:
            state.pending_base_count += 1
        if state.pending_base_count >= self.config.act_confirm_frames:
            state.base_closed = requested_closed
            state.pending_base_count = 0

    def _confirmed_class(
        self,
        state: _SideState,
        raw_class: int,
        confidence: float,
    ) -> GripperResidualClass:
        try:
            requested = GripperResidualClass(int(raw_class))
        except ValueError:
            requested = GripperResidualClass.KEEP_BASE
        if requested != GripperResidualClass.KEEP_BASE and confidence < self.config.residual_confidence_threshold:
            requested = GripperResidualClass.KEEP_BASE
        if requested != state.pending_class:
            state.pending_class = requested
            state.pending_count = 1
        else:
            state.pending_count += 1
        if state.pending_count >= self.config.residual_confirm_frames:
            state.active_class = requested
        return state.active_class

    def step(
        self,
        act_gripper: np.ndarray,
        residual_classes: np.ndarray | None,
        residual_confidences: np.ndarray | None,
        *,
        now_s: float,
    ) -> GripperControlResult:
        scores = np.asarray(act_gripper, dtype=np.float32).reshape(2)
        raw_classes = (
            np.zeros(2, dtype=np.int64)
            if residual_classes is None
            else np.asarray(residual_classes, dtype=np.int64).reshape(2)
        )
        confidences = (
            np.ones(2, dtype=np.float32)
            if residual_confidences is None
            else np.asarray(residual_confidences, dtype=np.float32).reshape(2)
        )
        confidences = np.where(np.isfinite(confidences), confidences, 0.0).astype(np.float32)
        candidate = np.empty(2, dtype=np.float32)
        final = np.empty(2, dtype=np.float32)
        base_closed = np.empty(2, dtype=np.bool_)
        for index, state in enumerate(self.sides):
            self._update_base(state, float(scores[index]))
            active = self._confirmed_class(
                state,
                int(raw_classes[index]),
                float(confidences[index]),
            )
            desired_closed = state.base_closed
            if active == GripperResidualClass.FORCE_OPEN:
                desired_closed = False
            elif active == GripperResidualClass.FORCE_CLOSE:
                desired_closed = True
            candidate[index] = self.config.close_value if desired_closed else self.config.open_value
            if desired_closed != state.final_closed and now_s - state.last_change_s >= self.config.min_hold_s:
                state.final_closed = desired_closed
                state.last_change_s = float(now_s)
            final[index] = self.config.close_value if state.final_closed else self.config.open_value
            base_closed[index] = state.base_closed
        return GripperControlResult(
            raw_classes=raw_classes,
            confidences=confidences,
            base_closed=base_closed,
            candidate_action=candidate,
            final_action=final,
        )
