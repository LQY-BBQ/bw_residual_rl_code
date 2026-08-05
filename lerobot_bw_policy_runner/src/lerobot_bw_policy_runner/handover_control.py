"""Bumpless REMOTE/INFERENCE handover for BW policy commands."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .config import HandoverConfig
from .constants import ARM_JOINT_INDICES, GRIPPER_JOINT_INDICES, JOINT_NAMES


class HandoverPhase(str, Enum):
    WAITING_FOR_SOURCE = "WAITING_FOR_SOURCE"
    REMOTE_SHADOW = "REMOTE_SHADOW"
    INITIAL_HOLD = "INITIAL_HOLD"
    RESUMING = "RESUMING"
    INFERENCE = "INFERENCE"


@dataclass(slots=True)
class HandoverResult:
    command: np.ndarray
    publish_control: bool
    phase: HandoverPhase
    target_error_max: float
    command_feedback_error_max: float
    reason: str | None = None


class PolicyHandoverController:
    """Gate and limit the command published on inactive/active Policy topics.

    Model inference is deliberately kept outside this class.  This class only
    decides whether a composed policy command may be published and, when it may,
    makes REMOTE -> INFERENCE transfer start at measured robot state.
    """

    def __init__(self, config: HandoverConfig, *, fps: float, gripper_hold_s: float) -> None:
        if fps <= 0:
            raise ValueError("handover fps must be positive")
        self.config = config
        self.fps = float(fps)
        self.gripper_hold_frames = max(1, int(math.ceil(max(float(gripper_hold_s), 0.0) * self.fps)))
        self.phase = HandoverPhase.WAITING_FOR_SOURCE
        self.control_source: int | None = None
        self.hold_frames_remaining = 0
        self.completion_count = 0
        self.frames_since_transition = 0
        self.previous_command: np.ndarray | None = None
        self.handover_gripper: np.ndarray | None = None
        self.requires_teleop_gripper = False

    @staticmethod
    def _vector(value: np.ndarray, *, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size != len(JOINT_NAMES):
            raise ValueError(f"{label} must have {len(JOINT_NAMES)} values, got {vector.size}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} contains NaN or Inf")
        return vector

    @staticmethod
    def _gripper(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        gripper = np.asarray(value, dtype=np.float32).reshape(-1)
        if gripper.size != len(GRIPPER_JOINT_INDICES) or not np.all(np.isfinite(gripper)):
            return None
        valid = np.isclose(gripper, 0.0, rtol=0.0, atol=1e-6) | np.isclose(
            gripper, 0.8, rtol=0.0, atol=1e-6
        )
        if not np.all(valid):
            return None
        return np.where(gripper >= 0.4, 0.8, 0.0).astype(np.float32)

    def observe_control_source(
        self,
        control_source: int | None,
        current_state: np.ndarray,
        teleop_gripper: np.ndarray | None,
    ) -> bool:
        """Update source state and return True when policy history must reset."""
        current = self._vector(current_state, label="current_state")
        if control_source not in (0, 1):
            self.phase = HandoverPhase.WAITING_FOR_SOURCE
            self.control_source = None
            self.previous_command = current.copy()
            self.handover_gripper = None
            self.requires_teleop_gripper = False
            return False

        source = int(control_source)
        previous_source = self.control_source
        if source == 0:
            self.phase = HandoverPhase.REMOTE_SHADOW
            self.control_source = 0
            self.previous_command = current.copy()
            self.hold_frames_remaining = 0
            self.completion_count = 0
            self.frames_since_transition = 0
            gripper = self._gripper(teleop_gripper)
            if gripper is not None:
                self.handover_gripper = gripper.copy()
            self.requires_teleop_gripper = True
            return False

        self.control_source = 1
        if previous_source != 1:
            self.phase = HandoverPhase.INITIAL_HOLD
            self.hold_frames_remaining = int(self.config.initial_hold_frames)
            self.completion_count = 0
            self.frames_since_transition = 0
            self.previous_command = current.copy()
            # Starting while control_source is already INFERENCE is still a
            # control-entry handover: require an explicit human gripper state
            # before publishing the first complete arm/gripper pair.
            self.requires_teleop_gripper = True
            gripper = self._gripper(teleop_gripper)
            self.handover_gripper = None if gripper is None else gripper.copy()
            return True
        return False

    def _hold_command(
        self,
        candidate: np.ndarray,
        current: np.ndarray,
        teleop_gripper: np.ndarray | None,
    ) -> tuple[np.ndarray, bool, str | None]:
        command = candidate.copy()
        command[ARM_JOINT_INDICES] = current[ARM_JOINT_INDICES]
        gripper = self._gripper(teleop_gripper)
        if gripper is not None:
            self.handover_gripper = gripper.copy()
        if self.handover_gripper is not None:
            command[GRIPPER_JOINT_INDICES] = self.handover_gripper
            return command, True, None
        if self.requires_teleop_gripper:
            return command, False, "waiting for a valid Teleop gripper command"
        return command, True, None

    def apply(
        self,
        candidate_action: np.ndarray,
        current_state: np.ndarray,
        teleop_gripper: np.ndarray | None,
    ) -> HandoverResult:
        candidate = self._vector(candidate_action, label="candidate_action")
        current = self._vector(current_state, label="current_state")

        if self.phase == HandoverPhase.WAITING_FOR_SOURCE:
            command = current.copy()
            command[GRIPPER_JOINT_INDICES] = candidate[GRIPPER_JOINT_INDICES]
            return self._result(
                command,
                candidate,
                current,
                publish=False,
                reason="waiting for control_source in {0,1}",
            )

        if self.phase == HandoverPhase.REMOTE_SHADOW:
            command, publish, reason = self._hold_command(candidate, current, teleop_gripper)
            self.previous_command = command.copy()
            return self._result(command, candidate, current, publish=publish, reason=reason)

        if self.phase == HandoverPhase.INITIAL_HOLD:
            command, publish, reason = self._hold_command(candidate, current, teleop_gripper)
            if not publish:
                return self._result(command, candidate, current, publish=False, reason=reason)
            self.previous_command = command.copy()
            self.hold_frames_remaining -= 1
            self.frames_since_transition += 1
            if self.hold_frames_remaining <= 0:
                self.phase = HandoverPhase.RESUMING
            return self._result(command, candidate, current, publish=True)

        if self.phase == HandoverPhase.RESUMING:
            if self.previous_command is None:
                self.previous_command = current.copy()
            max_step = float(self.config.resume_max_velocity) / self.fps
            previous_arm = self.previous_command[ARM_JOINT_INDICES]
            target_arm = candidate[ARM_JOINT_INDICES]
            command_arm = previous_arm + np.clip(target_arm - previous_arm, -max_step, max_step)
            tracking = float(self.config.max_command_tracking_error)
            current_arm = current[ARM_JOINT_INDICES]
            command_arm = np.clip(command_arm, current_arm - tracking, current_arm + tracking)
            command = candidate.copy()
            command[ARM_JOINT_INDICES] = command_arm
            if (
                self.handover_gripper is not None
                and self.frames_since_transition < self.gripper_hold_frames
            ):
                command[GRIPPER_JOINT_INDICES] = self.handover_gripper
            self.frames_since_transition += 1
            self.previous_command = command.copy()
            if float(np.max(np.abs(target_arm - command_arm))) <= float(self.config.completion_tolerance):
                self.completion_count += 1
            else:
                self.completion_count = 0
            if self.completion_count >= int(self.config.completion_frames):
                self.phase = HandoverPhase.INFERENCE
            return self._result(command, candidate, current, publish=True)

        self.previous_command = candidate.copy()
        return self._result(candidate, candidate, current, publish=True)

    def _result(
        self,
        command: np.ndarray,
        candidate: np.ndarray,
        current: np.ndarray,
        *,
        publish: bool,
        reason: str | None = None,
    ) -> HandoverResult:
        target_error = float(
            np.max(np.abs(candidate[ARM_JOINT_INDICES] - command[ARM_JOINT_INDICES]))
        )
        feedback_error = float(
            np.max(np.abs(command[ARM_JOINT_INDICES] - current[ARM_JOINT_INDICES]))
        )
        return HandoverResult(
            command=command.astype(np.float32, copy=True),
            publish_control=bool(publish),
            phase=self.phase,
            target_error_max=target_error,
            command_feedback_error_max=feedback_error,
            reason=reason,
        )
