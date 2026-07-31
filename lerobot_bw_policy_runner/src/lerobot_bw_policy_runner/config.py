"""YAML and command-line configuration for ACT / ACT+residual BC/RL inference."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml

from .constants import CAMERA_NAMES, CAMERA_SOURCES, CAMERA_TOPICS, JOINT_NAMES


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
    debug_gripper_residual_class: str


@dataclass(slots=True)
class RobotConfig:
    robot_sn: str
    input_topics: InputTopics
    output_topics: OutputTopics


@dataclass(slots=True)
class CameraSourceConfig:
    width: int
    height: int
    encoding: str


@dataclass(slots=True)
class CameraStreamConfig:
    sources: dict[str, CameraSourceConfig]
    expected_fps: float = 30.0
    minimum_fps: float = 28.5
    rate_measurement_s: float = 2.0
    max_frame_age_s: float = 0.15
    require_new_frames: bool = True


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
class GripperHysteresisConfig:
    enabled: bool = True
    open_threshold: float = 0.20
    close_threshold: float = 0.40
    single_threshold: float = 0.30


@dataclass(slots=True)
class GripperControlConfig:
    open_value: float = 0.0
    close_value: float = 0.8
    residual_confidence_threshold: float = 0.70
    residual_confirm_frames: int = 3
    min_hold_s: float = 0.30
    hysteresis: GripperHysteresisConfig = field(default_factory=GripperHysteresisConfig)


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
    camera_stream: CameraStreamConfig | None = None
    gripper: GripperControlConfig = field(default_factory=GripperControlConfig)


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


def load_config(config_path: str | Path, *, robot_sn: str | None = None, policy_path: str | Path | None = None, residual_policy_path: str | Path | None = None, mode: str | None = None, residual_lambda: float | None = None, device: str | None = None, fps: float | None = None, dry_run: bool | None = None, task: str | None = None, log_dir: str | Path | None = None, gripper_hysteresis: bool | None = None) -> AppConfig:
    raw = _read_yaml(Path(config_path))
    raw_ros = raw.get("ros", {}) or {}
    raw_robot = raw.get("robot", {}) or {}
    raw_inference = raw.get("inference", {}) or {}
    raw_residual = raw_inference.get("residual", {}) or {}
    raw_camera_stream = raw_inference.get("camera_stream", {}) or {}
    raw_gripper = raw_inference.get("gripper", {}) or {}
    raw_gripper_hysteresis = raw_gripper.get("hysteresis", {}) or {}

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
    camera_topics = {str(name): str(topic) for name, topic in camera_topics.items()}
    if tuple(camera_topics) != CAMERA_NAMES or camera_topics != CAMERA_TOPICS:
        raise ValueError(
            "robot.input_topics.cameras must match the ordered third-generation BW contract: "
            f"{CAMERA_TOPICS}, got {camera_topics}"
        )
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

    raw_camera_sources = raw_camera_stream.get("sources", {}) or {}
    if not isinstance(raw_camera_sources, dict):
        raise ValueError("inference.camera_stream.sources must be a mapping")
    camera_sources: dict[str, CameraSourceConfig] = {}
    for camera_name, source in raw_camera_sources.items():
        if camera_name not in camera_topics:
            raise ValueError(f"camera_stream.sources contains unknown camera {camera_name!r}")
        if not isinstance(source, dict):
            raise ValueError(f"camera_stream.sources.{camera_name} must be a mapping")
        width = int(source.get("width", 0))
        height = int(source.get("height", 0))
        encoding = str(source.get("encoding", "")).strip().lower()
        if width <= 0 or height <= 0 or not encoding:
            raise ValueError(
                f"camera_stream.sources.{camera_name} requires positive width/height and an encoding"
            )
        camera_sources[str(camera_name)] = CameraSourceConfig(width, height, encoding)
    missing_camera_sources = sorted(set(camera_topics) - set(camera_sources))
    if missing_camera_sources:
        raise ValueError(
            f"inference.camera_stream.sources is missing configured cameras: {missing_camera_sources}"
        )
    configured_sources = {
        name: (source.width, source.height, source.encoding)
        for name, source in camera_sources.items()
    }
    if tuple(configured_sources) != CAMERA_NAMES or configured_sources != CAMERA_SOURCES:
        raise ValueError(
            "inference.camera_stream.sources must match the ordered third-generation BW contract: "
            f"{CAMERA_SOURCES}, got {configured_sources}"
        )
    expected_camera_fps = float(raw_camera_stream.get("expected_fps", 30.0))
    minimum_camera_fps = float(raw_camera_stream.get("minimum_fps", expected_camera_fps * 0.95))
    rate_measurement_s = float(raw_camera_stream.get("rate_measurement_s", 2.0))
    max_frame_age_s = float(raw_camera_stream.get("max_frame_age_s", 0.15))
    if expected_camera_fps <= 0 or not 0 < minimum_camera_fps <= expected_camera_fps:
        raise ValueError("camera_stream expected_fps/minimum_fps must satisfy 0 < minimum <= expected")
    if rate_measurement_s <= 0 or max_frame_age_s <= 0:
        raise ValueError("camera_stream rate_measurement_s and max_frame_age_s must be positive")
    final_inference_fps = float(fps if fps is not None else raw_inference.get("fps", 30.0))
    if abs(final_inference_fps - expected_camera_fps) > 1e-6:
        raise ValueError(
            f"inference.fps={final_inference_fps:g} must match "
            f"inference.camera_stream.expected_fps={expected_camera_fps:g}"
        )

    smoothing_raw = raw_inference.get("smoothing", {}) or {}
    clamp_raw = raw_inference.get("clamp", {}) or {}
    gripper_name_style = str(raw_inference.get("gripper_name_style", "joint")).strip().lower()
    if gripper_name_style not in {"joint", "short"}:
        raise ValueError("inference.gripper_name_style must be 'joint' or 'short'")

    hysteresis_config = GripperHysteresisConfig(
        enabled=(
            bool(gripper_hysteresis)
            if gripper_hysteresis is not None
            else _as_bool(raw_gripper_hysteresis.get("enabled", True))
        ),
        open_threshold=float(raw_gripper_hysteresis.get("open_threshold", 0.20)),
        close_threshold=float(raw_gripper_hysteresis.get("close_threshold", 0.40)),
        single_threshold=float(raw_gripper_hysteresis.get("single_threshold", 0.30)),
    )
    gripper_config = GripperControlConfig(
        open_value=float(raw_gripper.get("open_value", 0.0)),
        close_value=float(raw_gripper.get("close_value", 0.8)),
        residual_confidence_threshold=float(raw_gripper.get("residual_confidence_threshold", 0.70)),
        residual_confirm_frames=int(raw_gripper.get("residual_confirm_frames", 3)),
        min_hold_s=float(raw_gripper.get("min_hold_s", 0.30)),
        hysteresis=hysteresis_config,
    )
    numeric_gripper_values = (
        gripper_config.open_value,
        gripper_config.close_value,
        gripper_config.residual_confidence_threshold,
        gripper_config.min_hold_s,
        hysteresis_config.open_threshold,
        hysteresis_config.single_threshold,
        hysteresis_config.close_threshold,
    )
    if not all(math.isfinite(value) for value in numeric_gripper_values):
        raise ValueError("inference.gripper values and thresholds must be finite")
    if gripper_config.open_value != 0.0 or gripper_config.close_value != 0.8:
        raise ValueError("inference.gripper command endpoints are fixed at open_value=0.0 and close_value=0.8")
    if not hysteresis_config.open_threshold < hysteresis_config.single_threshold < hysteresis_config.close_threshold:
        raise ValueError(
            "gripper thresholds must satisfy open_threshold < single_threshold < close_threshold"
        )
    if not (
        gripper_config.open_value
        <= hysteresis_config.open_threshold
        < hysteresis_config.close_threshold
        <= gripper_config.close_value
    ):
        raise ValueError("inference.gripper hysteresis thresholds must be within [open_value, close_value]")
    if not 0.0 <= gripper_config.residual_confidence_threshold <= 1.0:
        raise ValueError("inference.gripper.residual_confidence_threshold must be in [0, 1]")
    if gripper_config.residual_confirm_frames < 1:
        raise ValueError("inference.gripper.residual_confirm_frames must be at least 1")
    if gripper_config.min_hold_s < 0:
        raise ValueError("inference.gripper.min_hold_s must be non-negative")
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
                cameras=camera_topics,
                control_source=_expand_robot_sn(raw_input_topics.get("control_source", f"/{final_robot_sn}/Teleop/control_source"), final_robot_sn),
            ),
            output_topics=OutputTopics(
                arm_action=_expand_robot_sn(raw_output_topics["arm_action"], final_robot_sn),
                gripper_action=_expand_robot_sn(raw_output_topics["gripper_action"], final_robot_sn),
                debug_action_act=_expand_robot_sn(raw_output_topics.get("debug_action_act", f"/{final_robot_sn}/Policy/debug/action_act"), final_robot_sn),
                debug_action_rl_delta=_expand_robot_sn(raw_output_topics.get("debug_action_rl_delta", f"/{final_robot_sn}/Policy/debug/action_rl_delta"), final_robot_sn),
                debug_action_composed=_expand_robot_sn(raw_output_topics.get("debug_action_composed", f"/{final_robot_sn}/Policy/debug/action_composed"), final_robot_sn),
                debug_action_final=_expand_robot_sn(raw_output_topics.get("debug_action_final", f"/{final_robot_sn}/Policy/debug/action_final"), final_robot_sn),
                debug_gripper_residual_class=_expand_robot_sn(
                    raw_output_topics.get(
                        "debug_gripper_residual_class",
                        f"/{final_robot_sn}/Policy/debug/gripper_residual_class",
                    ),
                    final_robot_sn,
                ),
            ),
        ),
        inference=InferenceConfig(
            mode=final_mode,
            policy_path=final_policy_path,
            device=str(device if device is not None else raw_inference.get("device", "cuda")),
            fps=final_inference_fps,
            use_amp=_as_bool(raw_inference.get("use_amp", False)),
            task=" ".join(str(task if task is not None else raw_inference.get("task", "")).split()),
            robot_type=str(raw_inference.get("robot_type", "bw_runtime")),
            warmup_timeout_s=float(raw_inference.get("warmup_timeout_s", 10.0)),
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
            camera_stream=CameraStreamConfig(
                sources=camera_sources,
                expected_fps=expected_camera_fps,
                minimum_fps=minimum_camera_fps,
                rate_measurement_s=rate_measurement_s,
                max_frame_age_s=max_frame_age_s,
                require_new_frames=_as_bool(raw_camera_stream.get("require_new_frames", True)),
            ),
            gripper=gripper_config,
        ),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
