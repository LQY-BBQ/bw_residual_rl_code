"""LeRobot dataset creation and frame writing for BC/RL modes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
from pathlib import Path
import re
from typing import Any

import numpy as np

from .config import AppConfig
from .constants import (
    ACTION_ACT_KEY,
    ACTION_EXECUTED_KEY,
    ACTION_HUMAN_KEY,
    ACTION_KEY,
    ACTION_RL_DELTA_KEY,
    CONTROL_SOURCE_KEY,
    DATASET_JOINT_FEATURE_NAMES,
    DEFAULT_ROBOT_TYPE,
    DONE_KEY,
    HAS_HUMAN_ACTION_KEY,
    IMAGE_KEY_PREFIX,
    IS_INTERVENTION_KEY,
    OBS_STATE_KEY,
    REWARD_KEY,
    SUCCESS_KEY,
    TIMESTAMP_DIFF_PREFIX,
)


@dataclass(slots=True)
class DatasetHandle:
    dataset: Any
    dataset_path: Path
    repo_id: str
    features: dict[str, dict]


def sanitize_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "session"


def make_session_name(task: str, *, explicit_name: str | None = None, mode: str = "bc", episode_type: str = "demo") -> str:
    if explicit_name:
        return sanitize_name(explicit_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_suffix = sanitize_name(task)[:40]
    return f"session_{timestamp}_{mode}_{episode_type}_{task_suffix}"


def repo_id_from_prefix(repo_prefix: str, session_name: str) -> str:
    namespace, name_prefix = repo_prefix.strip("/").split("/", 1)
    return f"{namespace}/{name_prefix}_{sanitize_name(session_name)}"


def _vector_feature(dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": (len(DATASET_JOINT_FEATURE_NAMES),), "names": list(DATASET_JOINT_FEATURE_NAMES)}


def _scalar_feature(dtype: str = "float32") -> dict[str, Any]:
    return {"dtype": dtype, "shape": (1,), "names": ["value"]}


def build_features(sample_images: dict[str, np.ndarray], *, use_videos: bool, mode: str = "bc") -> dict[str, dict]:
    features: dict[str, dict] = {OBS_STATE_KEY: _vector_feature(), ACTION_KEY: _vector_feature()}
    if mode == "rl":
        features.update(
            {
                CONTROL_SOURCE_KEY: _scalar_feature("int64"),
                IS_INTERVENTION_KEY: _scalar_feature("int64"),
                HAS_HUMAN_ACTION_KEY: _scalar_feature("int64"),
                ACTION_ACT_KEY: _vector_feature(),
                ACTION_RL_DELTA_KEY: _vector_feature(),
                ACTION_HUMAN_KEY: _vector_feature(),
                ACTION_EXECUTED_KEY: _vector_feature(),
                REWARD_KEY: _scalar_feature(),
                DONE_KEY: _scalar_feature("int64"),
                SUCCESS_KEY: _scalar_feature("int64"),
                f"{TIMESTAMP_DIFF_PREFIX}.arm_action_dt": _scalar_feature(),
                f"{TIMESTAMP_DIFF_PREFIX}.gripper_action_dt": _scalar_feature(),
                f"{TIMESTAMP_DIFF_PREFIX}.action_act_dt": _scalar_feature(),
                f"{TIMESTAMP_DIFF_PREFIX}.action_final_dt": _scalar_feature(),
            }
        )
    image_dtype = "video" if use_videos else "image"
    for camera_name, image in sample_images.items():
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Camera {camera_name!r} must be RGB HWC, got {getattr(image, 'shape', None)}")
        features[f"{IMAGE_KEY_PREFIX}.{camera_name}"] = {
            "dtype": image_dtype,
            "shape": tuple(int(v) for v in image.shape),
            "names": ["height", "width", "channels"],
        }
    return features


def _vec(value: Any, *, zeros: bool = False) -> np.ndarray:
    if value is None and zeros:
        return np.zeros((len(DATASET_JOINT_FEATURE_NAMES),), dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(len(DATASET_JOINT_FEATURE_NAMES))


def _scalar(value: Any, dtype=np.float32) -> np.ndarray:
    return np.asarray([value], dtype=dtype)


def build_frame(
    sample: Any,
    task: str,
    *,
    mode: str = "bc",
    reward: float = 0.0,
    done: bool = False,
    success: bool = False,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        OBS_STATE_KEY: np.asarray(sample.observation_state, dtype=np.float32),
        ACTION_KEY: np.asarray(sample.action, dtype=np.float32),
        "task": task,
    }
    if mode == "rl":
        timing = sample.timing or {}
        frame.update(
            {
                CONTROL_SOURCE_KEY: _scalar(sample.control_source if sample.control_source is not None else -1, np.int64),
                IS_INTERVENTION_KEY: _scalar(1 if sample.is_intervention else 0, np.int64),
                HAS_HUMAN_ACTION_KEY: _scalar(1 if sample.has_human_action else 0, np.int64),
                ACTION_ACT_KEY: _vec(sample.action_act, zeros=True),
                ACTION_RL_DELTA_KEY: _vec(sample.action_rl_delta, zeros=True),
                ACTION_HUMAN_KEY: _vec(sample.action_human, zeros=True),
                ACTION_EXECUTED_KEY: _vec(sample.action_executed, zeros=True),
                REWARD_KEY: _scalar(float(reward), np.float32),
                DONE_KEY: _scalar(1 if done else 0, np.int64),
                SUCCESS_KEY: _scalar(1 if success else 0, np.int64),
                f"{TIMESTAMP_DIFF_PREFIX}.arm_action_dt": _scalar(float(timing.get("arm_action_dt", 0.0)), np.float32),
                f"{TIMESTAMP_DIFF_PREFIX}.gripper_action_dt": _scalar(float(timing.get("gripper_action_dt", 0.0)), np.float32),
                f"{TIMESTAMP_DIFF_PREFIX}.action_act_dt": _scalar(float(timing.get("action_act_dt", 0.0)), np.float32),
                f"{TIMESTAMP_DIFF_PREFIX}.action_final_dt": _scalar(float(timing.get("action_final_dt", 0.0)), np.float32),
            }
        )
    for camera_name, image in sample.images.items():
        frame[f"{IMAGE_KEY_PREFIX}.{camera_name}"] = np.ascontiguousarray(image, dtype=np.uint8)
    return frame


def create_lerobot_dataset(config: AppConfig, sample: Any, *, session_name: str | None = None, overwrite: bool = False) -> DatasetHandle:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    final_session_name = make_session_name(
        config.dataset.task,
        explicit_name=session_name,
        mode=config.dataset.mode,
        episode_type=config.dataset.episode_type,
    )
    dataset_path = config.dataset.root.expanduser() / final_session_name
    if dataset_path.exists() and not overwrite:
        raise FileExistsError(f"Dataset path already exists: {dataset_path}. Use --session-name new name or --overwrite.")
    if dataset_path.exists() and overwrite:
        import shutil
        shutil.rmtree(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    repo_id = repo_id_from_prefix(config.dataset.repo_prefix, final_session_name)
    features = build_features(sample.images, use_videos=config.dataset.use_videos, mode=config.dataset.mode)

    kwargs = {
        "repo_id": repo_id,
        "fps": int(config.dataset.fps),
        "features": features,
        "root": dataset_path,
        "robot_type": DEFAULT_ROBOT_TYPE,
        "use_videos": bool(config.dataset.use_videos),
    }
    signature = inspect.signature(LeRobotDataset.create)
    if "image_writer_processes" in signature.parameters:
        kwargs["image_writer_processes"] = 0
    if "image_writer_threads" in signature.parameters:
        kwargs["image_writer_threads"] = 12 if config.dataset.fps < 25 else 15
    if "batch_encoding_size" in signature.parameters:
        kwargs["batch_encoding_size"] = 1
    if "video_codec" in signature.parameters:
        kwargs["video_codec"] = config.dataset.video_codec
    if "vcodec" in signature.parameters:
        kwargs["vcodec"] = config.dataset.video_codec

    dataset = LeRobotDataset.create(**kwargs)
    return DatasetHandle(dataset=dataset, dataset_path=dataset_path, repo_id=repo_id, features=features)


def finalize_dataset(dataset: Any) -> None:
    if hasattr(dataset, "finalize"):
        dataset.finalize()
