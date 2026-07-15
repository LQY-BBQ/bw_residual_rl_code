"""YAML and command-line configuration for ACT / ACT+residual BC/RL inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .constants import JOINT_NAMES


@dataclass(slots=True)
class RosConfig:
    domain_id: int = 0


@dataclass(slots=True)
class InputTopics:
    state: str
    cameras: dict[str, str]
    control_source: str | None = None


@dataclass(slots=True)
class OutputTopics:
    arm_action: str
    gripper_action: str
    debug_action_act: str
    debug_action_rl_delta: str
    debug_action_composed: str
    debug_action_final: str


@dataclass(slots=True)
class RobotConfig:
    robot_sn: str
    input_topics: InputTopics
    output_topics: OutputTopics


@dataclass(slots=True)
class ActionSmoothingConfig:
    enabled: bool = False
    alpha: float = 1.0
    max_delta: float | None = None
    gripper_max_delta: float | None = None


@dataclass(slots=True)
class ActionClampConfig:
    enabled: bool = False
    min: list[float] | None = None
    max: list[float] | None = None


@dataclass(slots=True)
class ResidualConfig:
    policy_path: Path | None = None
    lambda_: float | None = None
    deterministic: bool = True
    limits: list[float] = field(default_factory=lambda: [0.03] * len(JOINT_NAMES))


@dataclass(slots=True)
class InferenceConfig:
    mode: str = "act"
    policy_path: Path | None = None
    device: str = "cuda"
    fps: float = 30.0
    use_amp: bool = False
    task: str = ""
    robot_type: str = "bw_runtime"
    warmup_timeout_s: float = 10.0
    resize_images_to_policy_shape: bool = True
    dry_run: bool = False
    reset_policy_on_start: bool = True
    require_all_cameras: bool = True
    arm_velocity_limit: float | None = None
    gripper_name_style: str = "joint"
    log_every_n_steps: int = 30
    smoothing: ActionSmoothingConfig | None = None
    clamp: ActionClampConfig | None = None
    residual: ResidualConfig | None = None
    log_dir: Path | None = None


@dataclass(slots=True)
class AppConfig:
    ros: RosConfig
    robot: RobotConfig
    inference: InferenceConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")
    return data


def _expand_robot_sn(text: str, robot_sn: str) -> str:
    return str(text).replace("{robot_sn}", robot_sn).replace("{robot_id}", robot_sn)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _as_optional_path(value: Any) -> Path | None:
    if value in {None, "", "null", "None"}:
        return None
    return Path(str(value)).expanduser()


def _as_optional_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Expected a list of floats or null, got {type(value).__name__}")
    return [float(v) for v in value]


def _residual_limits(raw: Any) -> list[float]:
    if raw is None:
        return [0.03] * len(JOINT_NAMES)
    if isinstance(raw, (float, int)):
        return [float(raw)] * len(JOINT_NAMES)
    if isinstance(raw, dict):
        out = []
        for name in JOINT_NAMES:
            if name in raw:
                out.append(float(raw[name]))
            elif name.endswith("gripper_joint") and "gripper" in raw:
                out.append(float(raw["gripper"]))
            elif "default" in raw:
                out.append(float(raw["default"]))
            else:
                out.append(0.03)
        return out
    if isinstance(raw, list):
        if len(raw) != len(JOINT_NAMES):
            raise ValueError(f"residual.limits list must have {len(JOINT_NAMES)} values")
        return [float(v) for v in raw]
    raise ValueError("residual.limits must be scalar, list, dict, or null")


def load_config(config_path: str | Path, *, robot_sn: str | None = None, policy_path: str | Path | None = None, residual_policy_path: str | Path | None = None, mode: str | None = None, residual_lambda: float | None = None, device: str | None = None, fps: float | None = None, dry_run: bool | None = None, task: str | None = None, log_dir: str | Path | None = None) -> AppConfig:
    raw = _read_yaml(Path(config_path))
    raw_ros = raw.get("ros", {}) or {}
    raw_robot = raw.get("robot", {}) or {}
    raw_inference = raw.get("inference", {}) or {}
    raw_residual = raw_inference.get("residual", {}) or {}

    final_robot_sn = str(robot_sn or raw_robot.get("robot_sn") or "").strip()
    if not final_robot_sn or final_robot_sn == "BW_XXXXXXX":
        raise ValueError("robot_sn is required. Pass --robot-sn BW_xxx or set robot.robot_sn in YAML.")

    raw_input_topics = raw_robot.get("input_topics", {}) or {}
    raw_output_topics = raw_robot.get("output_topics", {}) or {}
    if "state" not in raw_input_topics:
        raise ValueError("Missing robot.input_topics.state")
    camera_topics = raw_input_topics.get("cameras", {}) or {}
    if not isinstance(camera_topics, dict) or not camera_topics:
        raise ValueError("robot.input_topics.cameras must be a non-empty mapping")
    for key in ["arm_action", "gripper_action"]:
        if key not in raw_output_topics:
            raise ValueError(f"Missing robot.output_topics.{key}")

    final_mode = str(mode or raw_inference.get("mode", "act")).strip().lower()
    if final_mode == "act_residual_sac":
        final_mode = "act_residual_rl"
    if final_mode not in {"act", "act_residual_rl", "act_residual_bc"}:
        raise ValueError("mode must be act, act_residual_bc, act_residual_rl, or alias act_residual_sac")

    final_policy_path = _as_optional_path(policy_path if policy_path is not None else raw_inference.get("policy_path"))
    final_residual_path = _as_optional_path(residual_policy_path if residual_policy_path is not None else raw_residual.get("policy_path"))
    final_log_dir = _as_optional_path(log_dir if log_dir is not None else raw_inference.get("log_dir"))

    smoothing_raw = raw_inference.get("smoothing", {}) or {}
    clamp_raw = raw_inference.get("clamp", {}) or {}
    gripper_name_style = str(raw_inference.get("gripper_name_style", "joint")).strip().lower()
    if gripper_name_style not in {"joint", "short"}:
        raise ValueError("inference.gripper_name_style must be 'joint' or 'short'")

    residual_cfg = ResidualConfig(
        policy_path=final_residual_path,
        lambda_=(
            float(residual_lambda)
            if residual_lambda is not None
            else _as_optional_float(raw_residual.get("lambda"))
        ),
        deterministic=_as_bool(raw_residual.get("deterministic", True)),
        limits=_residual_limits(raw_residual.get("limits")),
    )
    return AppConfig(
        ros=RosConfig(domain_id=int(raw_ros.get("domain_id", 0))),
        robot=RobotConfig(
            robot_sn=final_robot_sn,
            input_topics=InputTopics(
                state=_expand_robot_sn(raw_input_topics["state"], final_robot_sn),
                cameras={str(name): str(topic) for name, topic in camera_topics.items()},
                control_source=_expand_robot_sn(raw_input_topics.get("control_source", f"/{final_robot_sn}/Teleop/control_source"), final_robot_sn),
            ),
            output_topics=OutputTopics(
                arm_action=_expand_robot_sn(raw_output_topics["arm_action"], final_robot_sn),
                gripper_action=_expand_robot_sn(raw_output_topics["gripper_action"], final_robot_sn),
                debug_action_act=_expand_robot_sn(raw_output_topics.get("debug_action_act", f"/{final_robot_sn}/Policy/debug/action_act"), final_robot_sn),
                debug_action_rl_delta=_expand_robot_sn(raw_output_topics.get("debug_action_rl_delta", f"/{final_robot_sn}/Policy/debug/action_rl_delta"), final_robot_sn),
                debug_action_composed=_expand_robot_sn(raw_output_topics.get("debug_action_composed", f"/{final_robot_sn}/Policy/debug/action_composed"), final_robot_sn),
                debug_action_final=_expand_robot_sn(raw_output_topics.get("debug_action_final", f"/{final_robot_sn}/Policy/debug/action_final"), final_robot_sn),
            ),
        ),
        inference=InferenceConfig(
            mode=final_mode,
            policy_path=final_policy_path,
            device=str(device if device is not None else raw_inference.get("device", "cuda")),
            fps=float(fps if fps is not None else raw_inference.get("fps", 30.0)),
            use_amp=_as_bool(raw_inference.get("use_amp", False)),
            task=" ".join(str(task if task is not None else raw_inference.get("task", "")).split()),
            robot_type=str(raw_inference.get("robot_type", "bw_runtime")),
            warmup_timeout_s=float(raw_inference.get("warmup_timeout_s", 10.0)),
            resize_images_to_policy_shape=_as_bool(raw_inference.get("resize_images_to_policy_shape", True)),
            dry_run=bool(dry_run) if dry_run is not None else _as_bool(raw_inference.get("dry_run", False)),
            reset_policy_on_start=_as_bool(raw_inference.get("reset_policy_on_start", True)),
            require_all_cameras=_as_bool(raw_inference.get("require_all_cameras", True)),
            arm_velocity_limit=_as_optional_float(raw_inference.get("arm_velocity_limit")),
            gripper_name_style=gripper_name_style,
            log_every_n_steps=int(raw_inference.get("log_every_n_steps", 30)),
            smoothing=ActionSmoothingConfig(enabled=_as_bool(smoothing_raw.get("enabled", False)), alpha=float(smoothing_raw.get("alpha", 1.0)), max_delta=_as_optional_float(smoothing_raw.get("max_delta")), gripper_max_delta=_as_optional_float(smoothing_raw.get("gripper_max_delta"))),
            clamp=ActionClampConfig(enabled=_as_bool(clamp_raw.get("enabled", False)), min=_as_optional_float_list(clamp_raw.get("min")), max=_as_optional_float_list(clamp_raw.get("max"))),
            residual=residual_cfg,
            log_dir=final_log_dir,
        ),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
