"""Configuration loading for BC/RL BW data collection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RosConfig:
    domain_id: int = 0


@dataclass(slots=True)
class RobotTopics:
    state: str
    arm_action: str
    gripper_action: str
    control_source: str | None = None
    action_act: str | None = None
    action_rl_delta: str | None = None
    action_final: str | None = None


@dataclass(slots=True)
class RobotConfig:
    robot_sn: str
    topics: RobotTopics


@dataclass(slots=True)
class CameraConfig:
    topics: dict[str, str]


@dataclass(slots=True)
class DatasetConfig:
    root: Path
    repo_prefix: str
    fps: int
    task: str
    use_videos: bool = True
    video_codec: str = "h264"
    mode: str = "bc"  # bc or rl
    episode_type: str = "demo"  # demo, correction, rollout, eval


@dataclass(slots=True)
class RecordConfig:
    warmup_timeout_s: float = 10.0
    require_all_topics: bool = True
    require_rl_debug_topics: bool = True
    log_every_n_frames: int = 30


@dataclass(slots=True)
class AppConfig:
    ros: RosConfig
    robot: RobotConfig
    cameras: CameraConfig
    dataset: DatasetConfig
    record: RecordConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a YAML mapping: {path}")
    return data


def _expand_robot_sn(text: str | None, robot_sn: str) -> str | None:
    if text is None:
        return None
    return str(text).replace("{robot_sn}", robot_sn).replace("{robot_id}", robot_sn)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_repo_prefix(repo_prefix: str) -> str:
    normalized = str(repo_prefix).strip().strip("/")
    if normalized.count("/") != 1:
        raise ValueError(f"dataset.repo_prefix must look like 'namespace/name_prefix', got {repo_prefix!r}")
    namespace, prefix = normalized.split("/", 1)
    if not namespace or not prefix or any(ch.isspace() for ch in normalized):
        raise ValueError(f"dataset.repo_prefix must contain no spaces, got {repo_prefix!r}")
    return normalized


def load_config(
    config_path: str | Path,
    *,
    robot_sn: str | None = None,
    dataset_root: str | Path | None = None,
    task: str | None = None,
    fps: int | None = None,
    mode: str | None = None,
    episode_type: str | None = None,
) -> AppConfig:
    raw = _read_yaml(Path(config_path))
    raw_ros = raw.get("ros", {}) or {}
    raw_robot = raw.get("robot", {}) or {}
    raw_cameras = raw.get("cameras", {}) or {}
    raw_dataset = raw.get("dataset", {}) or {}
    raw_record = raw.get("record", {}) or {}

    final_robot_sn = str(robot_sn or raw_robot.get("robot_sn") or "").strip()
    if not final_robot_sn or final_robot_sn == "BW_XXXXXXX":
        raise ValueError("robot_sn is required. Pass --robot-sn BW_xxx or set robot.robot_sn in YAML.")

    final_mode = str(mode or raw_dataset.get("mode", "bc")).strip().lower()
    if final_mode not in {"bc", "rl"}:
        raise ValueError("dataset.mode / --mode must be 'bc' or 'rl'")
    final_episode_type = str(episode_type or raw_dataset.get("episode_type", "demo")).strip().lower()

    raw_topics = raw_robot.get("topics", {}) or {}
    for key in ["state", "arm_action", "gripper_action"]:
        if key not in raw_topics:
            raise ValueError(f"Missing robot.topics.{key}")

    robot_topics = RobotTopics(
        state=_expand_robot_sn(raw_topics["state"], final_robot_sn) or "",
        arm_action=_expand_robot_sn(raw_topics["arm_action"], final_robot_sn) or "",
        gripper_action=_expand_robot_sn(raw_topics["gripper_action"], final_robot_sn) or "",
        control_source=_expand_robot_sn(raw_topics.get("control_source"), final_robot_sn),
        action_act=_expand_robot_sn(raw_topics.get("action_act"), final_robot_sn),
        action_rl_delta=_expand_robot_sn(raw_topics.get("action_rl_delta"), final_robot_sn),
        action_final=_expand_robot_sn(raw_topics.get("action_final"), final_robot_sn),
    )
    if final_mode == "rl":
        for key in ["control_source", "action_act", "action_rl_delta", "action_final"]:
            if getattr(robot_topics, key) in {None, ""}:
                raise ValueError(f"RL mode requires robot.topics.{key}")

    camera_topics = raw_cameras.get("topics", {}) or {}
    if not isinstance(camera_topics, dict) or not camera_topics:
        raise ValueError("cameras.topics must be a non-empty mapping")
    camera_topics = {str(name): str(topic) for name, topic in camera_topics.items()}

    dataset_root_path = Path(dataset_root or raw_dataset.get("root", "~/robot_datasets/bw_lerobot")).expanduser()
    final_task = " ".join(str(task or raw_dataset.get("task", "")).split())
    if not final_task:
        raise ValueError("dataset.task cannot be empty. Pass --task or set dataset.task in YAML.")
    final_fps = int(fps if fps is not None else raw_dataset.get("fps", 30))
    if final_fps <= 0:
        raise ValueError(f"dataset.fps must be positive, got {final_fps}")

    return AppConfig(
        ros=RosConfig(domain_id=int(raw_ros.get("domain_id", 0))),
        robot=RobotConfig(robot_sn=final_robot_sn, topics=robot_topics),
        cameras=CameraConfig(topics=camera_topics),
        dataset=DatasetConfig(
            root=dataset_root_path,
            repo_prefix=_normalize_repo_prefix(str(raw_dataset.get("repo_prefix", "local/bw_mantis"))),
            fps=final_fps,
            task=final_task,
            use_videos=_as_bool(raw_dataset.get("use_videos", True)),
            video_codec=str(raw_dataset.get("video_codec", "h264")),
            mode=final_mode,
            episode_type=final_episode_type,
        ),
        record=RecordConfig(
            warmup_timeout_s=float(raw_record.get("warmup_timeout_s", 10.0)),
            require_all_topics=_as_bool(raw_record.get("require_all_topics", True)),
            require_rl_debug_topics=_as_bool(raw_record.get("require_rl_debug_topics", True)),
            log_every_n_frames=int(raw_record.get("log_every_n_frames", 30)),
        ),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
