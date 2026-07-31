"""Action validation, safety limiting, residual composition, and ROS message splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable

import numpy as np
from sensor_msgs.msg import JointState

from .config import ActionClampConfig, ActionSmoothingConfig, GripperControlConfig
from .constants import (
    ARM_JOINT_INDICES,
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_INDICES,
    GRIPPER_JOINT_NAMES,
    GRIPPER_SHORT_NAMES,
    JOINT_NAMES,
)


@dataclass(slots=True)
class ActionFilterState:
    previous_action: np.ndarray | None = None


def align_action_vector(action: np.ndarray, current_state: np.ndarray, *, expected_dim: int = 16) -> np.ndarray:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    if flat.size < expected_dim:
        filled = np.asarray(current_state, dtype=np.float32).reshape(-1).copy()
        if filled.size != expected_dim:
            raise ValueError(f"current_state must have {expected_dim} values, got {filled.size}")
        filled[: flat.size] = flat
        flat = filled
    elif flat.size > expected_dim:
        flat = flat[:expected_dim]
    if flat.size != expected_dim:
        raise ValueError(f"Policy action must have {expected_dim} values, got {flat.size}")
    if not np.all(np.isfinite(flat)):
        raise ValueError("Policy action contains NaN or Inf")
    return np.asarray(flat, dtype=np.float32)


def apply_clamp(action: np.ndarray, clamp: ActionClampConfig | None) -> np.ndarray:
    if clamp is None or not clamp.enabled:
        return action
    out = np.asarray(action, dtype=np.float32).copy()
    if clamp.min is not None:
        if len(clamp.min) != out.size:
            raise ValueError(f"clamp.min must have {out.size} values, got {len(clamp.min)}")
        out = np.maximum(out, np.asarray(clamp.min, dtype=np.float32))
    if clamp.max is not None:
        if len(clamp.max) != out.size:
            raise ValueError(f"clamp.max must have {out.size} values, got {len(clamp.max)}")
        out = np.minimum(out, np.asarray(clamp.max, dtype=np.float32))
    return out


def apply_smoothing(action: np.ndarray, current_state: np.ndarray, smoothing: ActionSmoothingConfig | None, state: ActionFilterState) -> np.ndarray:
    if smoothing is None or not smoothing.enabled:
        state.previous_action = np.asarray(action, dtype=np.float32).copy()
        return action
    target = np.asarray(action, dtype=np.float32)
    baseline = state.previous_action
    if baseline is None:
        baseline = np.asarray(current_state, dtype=np.float32)
    alpha = float(np.clip(smoothing.alpha, 0.0, 1.0))
    out = baseline + alpha * (target - baseline)
    if smoothing.max_delta is not None:
        max_delta = abs(float(smoothing.max_delta))
        out = baseline + np.clip(out - baseline, -max_delta, max_delta)
    if smoothing.gripper_max_delta is not None:
        max_delta = abs(float(smoothing.gripper_max_delta))
        for index in (JOINT_NAMES.index("left_gripper_joint"), JOINT_NAMES.index("right_gripper_joint")):
            out[index] = baseline[index] + float(np.clip(out[index] - baseline[index], -max_delta, max_delta))
    state.previous_action = out.astype(np.float32, copy=True)
    return state.previous_action


def compose_residual_action(action_act: np.ndarray, delta_norm_or_joint: np.ndarray, *, residual_limits: Iterable[float], residual_lambda: float, delta_is_normalized: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_joint, final_action_raw) before clamp/smoothing.

    SAC checkpoints store normalized residuals in [-1, 1]. During deployment,
    normalized residuals are converted to joint-position corrections with configured limits.
    """
    base = np.asarray(action_act, dtype=np.float32).reshape(len(JOINT_NAMES))
    raw_delta = np.asarray(delta_norm_or_joint, dtype=np.float32).reshape(-1)
    raw_limits = np.asarray(list(residual_limits), dtype=np.float32).reshape(-1)
    if raw_delta.size != len(ARM_JOINT_INDICES):
        raise ValueError(f"Hybrid residual action must have 14 arm values, got {raw_delta.size}")
    if raw_limits.size != len(ARM_JOINT_INDICES):
        raise ValueError(f"14-D arm residual requires 14 limits, got {raw_limits.size}")
    delta = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    limits = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    delta[ARM_JOINT_INDICES] = raw_delta
    limits[ARM_JOINT_INDICES] = raw_limits
    if delta_is_normalized:
        delta = np.clip(delta, -1.0, 1.0) * limits
    else:
        delta = np.clip(delta, -np.abs(limits), np.abs(limits))
    final = base + float(residual_lambda) * delta
    final[GRIPPER_JOINT_INDICES] = base[GRIPPER_JOINT_INDICES]
    return delta.astype(np.float32), final.astype(np.float32)


def vector_to_joint_state(action: np.ndarray, *, stamp, name_style: str = "joint") -> JointState:
    values = np.asarray(action, dtype=np.float32).reshape(len(JOINT_NAMES))
    msg = JointState()
    msg.header.stamp = stamp
    msg.name = list(JOINT_NAMES)
    msg.position = [float(v) for v in values]
    return msg


def split_action_to_joint_states(action: np.ndarray, *, stamp, gripper_name_style: str = "joint", arm_velocity_limit: float | None = None) -> tuple[JointState, JointState]:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size != len(JOINT_NAMES):
        raise ValueError(f"Expected {len(JOINT_NAMES)} action values, got {values.size}")
    value_by_name = {name: float(values[index]) for index, name in enumerate(JOINT_NAMES)}
    arm_msg = JointState()
    arm_msg.header.stamp = stamp
    arm_msg.name = list(ARM_JOINT_NAMES)
    arm_msg.position = [value_by_name[name] for name in ARM_JOINT_NAMES]
    if arm_velocity_limit is not None:
        arm_msg.velocity = [float(arm_velocity_limit)] * len(ARM_JOINT_NAMES)
        arm_msg.effort = [0.0] * len(ARM_JOINT_NAMES)
    gripper_msg = JointState()
    gripper_msg.header.stamp = stamp
    gripper_msg.name = list(GRIPPER_SHORT_NAMES if gripper_name_style == "short" else GRIPPER_JOINT_NAMES)
    gripper_msg.position = [value_by_name["left_gripper_joint"], value_by_name["right_gripper_joint"]]
    return arm_msg, gripper_msg


class ActionCSVLogger:
    """Optional local CSV debug logger for runner outputs."""

    def __init__(self, log_dir: Path | None, *, gripper_config: GripperControlConfig | None = None) -> None:
        self.log_dir = Path(log_dir).expanduser() if log_dir is not None else None
        self.gripper_config = gripper_config
        self._file = None
        self._writer = None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._file = (self.log_dir / "policy_actions.csv").open("w", newline="", encoding="utf-8")
            fieldnames = ["step", "control_source"]
            for prefix in ["act", "delta", "final"]:
                fieldnames.extend(f"{prefix}.{name}" for name in JOINT_NAMES)
            fieldnames.extend(
                [
                    "gripper_class.left",
                    "gripper_class.right",
                    "gripper_confidence.left",
                    "gripper_confidence.right",
                    "gripper_hysteresis_enabled",
                    "gripper_open_value",
                    "gripper_close_value",
                    "gripper_residual_confidence_threshold",
                    "gripper_residual_confirm_frames",
                    "gripper_min_hold_s",
                    "gripper_open_threshold",
                    "gripper_single_threshold",
                    "gripper_close_threshold",
                ]
            )
            self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
            self._writer.writeheader()

    def write(self, *, step: int, control_source: int | None, action_act: np.ndarray, delta: np.ndarray, action_final: np.ndarray, gripper_classes: np.ndarray | None = None, gripper_confidences: np.ndarray | None = None, gripper_hysteresis_enabled: bool | None = None) -> None:
        if self._writer is None:
            return
        row = {"step": step, "control_source": -1 if control_source is None else int(control_source)}
        for prefix, values in [("act", action_act), ("delta", delta), ("final", action_final)]:
            for name, value in zip(JOINT_NAMES, np.asarray(values).reshape(len(JOINT_NAMES))):
                row[f"{prefix}.{name}"] = float(value)
        classes = np.zeros(2, dtype=np.int64) if gripper_classes is None else np.asarray(gripper_classes).reshape(2)
        confidences = np.ones(2, dtype=np.float32) if gripper_confidences is None else np.asarray(gripper_confidences).reshape(2)
        row["gripper_class.left"], row["gripper_class.right"] = (int(v) for v in classes)
        row["gripper_confidence.left"], row["gripper_confidence.right"] = (float(v) for v in confidences)
        row["gripper_hysteresis_enabled"] = int(bool(gripper_hysteresis_enabled))
        config = self.gripper_config
        if config is not None:
            row.update(
                {
                    "gripper_open_value": config.open_value,
                    "gripper_close_value": config.close_value,
                    "gripper_residual_confidence_threshold": config.residual_confidence_threshold,
                    "gripper_residual_confirm_frames": config.residual_confirm_frames,
                    "gripper_min_hold_s": config.min_hold_s,
                    "gripper_open_threshold": config.hysteresis.open_threshold,
                    "gripper_single_threshold": config.hysteresis.single_threshold,
                    "gripper_close_threshold": config.hysteresis.close_threshold,
                }
            )
        self._writer.writerow(row)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
