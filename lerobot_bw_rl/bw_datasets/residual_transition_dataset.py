"""Visual residual BC and residual SAC datasets built from BW RL recordings."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from visual_cache import VisualCache, load_visual_feature_cache

JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_gripper_joint",
    "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_gripper_joint",
]
ARM_INDICES = np.asarray([index for index, name in enumerate(JOINT_NAMES) if "gripper" not in name], dtype=np.int64)
GRIPPER_INDICES = np.asarray([JOINT_NAMES.index("left_gripper_joint"), JOINT_NAMES.index("right_gripper_joint")], dtype=np.int64)
GRIPPER_CLASS_NAMES = ("KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE")
GRIPPER_THRESHOLD_EPSILON = 1e-6


def _as_vec(x: Any, dim: int = 16) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"Expected vector dim={dim}, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Vector contains NaN or Inf")
    return arr


def _as_scalar(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    arr = np.asarray(x).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def read_lerobot_parquets(root: str | Path) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    if "index" in df.columns:
        recorded = np.asarray([int(_as_scalar(v)) for v in df["index"]], dtype=np.int64)
        expected = np.arange(len(df), dtype=np.int64)
        if not np.array_equal(recorded, expected):
            raise ValueError("Dataset global index column is not contiguous; visual cache alignment is unsafe.")
    return df


def read_reward_column(df: pd.DataFrame) -> np.ndarray:
    if "reward" not in df.columns:
        raise ValueError("Dataset is missing reward. Record with data_collector --mode rl.")
    values = np.asarray([_as_scalar(v, 0.0) for v in df["reward"]], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Dataset reward contains NaN or Inf")
    return values


def compute_done(df: pd.DataFrame) -> np.ndarray:
    if "done" in df.columns:
        values = np.asarray([_as_scalar(v, 0.0) for v in df["done"]], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Dataset done contains NaN or Inf")
        return (values >= 0.5).astype(np.float32)
    done = np.zeros(len(df), dtype=np.float32)
    if "episode_index" in df.columns:
        episode = np.asarray([int(_as_scalar(v)) for v in df["episode_index"]], dtype=np.int64)
        done[:-1] = episode[:-1] != episode[1:]
    done[-1] = 1.0
    return done


@dataclass(slots=True)
class ObservationStats:
    mean: np.ndarray
    std: np.ndarray
    clip: float = 10.0

    def normalize(self, value: np.ndarray) -> np.ndarray:
        output = (np.asarray(value, dtype=np.float32) - self.mean) / self.std
        if self.clip > 0:
            output = np.clip(output, -self.clip, self.clip)
        return output.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": float(self.clip)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ObservationStats":
        mean = np.asarray(raw["mean"], dtype=np.float32)
        std = np.asarray(raw["std"], dtype=np.float32)
        if mean.shape != std.shape or mean.ndim != 1:
            raise ValueError("Invalid observation normalization metadata")
        return cls(mean=mean, std=np.maximum(std, 1e-6), clip=float(raw.get("clip", 10.0)))


@dataclass(slots=True)
class ResidualDatasetConfig:
    root: Path
    residual_limits: np.ndarray
    residual_lambda: float = 0.2
    visual_cache: VisualCache | str | Path | None = None
    observation_stats: ObservationStats | None = None
    normalization_clip: float = 10.0
    use_only_interventions: bool = False
    gripper_hysteresis_enabled: bool = True
    gripper_open_threshold: float = 0.50
    gripper_close_threshold: float = 0.40
    gripper_single_threshold: float = 0.45
    gripper_act_confirm_frames: int = 3


def preflight_gripper_event_counts(cfg: ResidualDatasetConfig, minimum: int) -> np.ndarray:
    """Validate correction coverage without building the expensive visual cache."""
    df = read_lerobot_parquets(cfg.root)
    required = ("action.act", "action.executed", "is_intervention")
    missing = [key for key in required if key not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing gripper-label fields: {missing}")
    action_act = np.stack([_as_vec(value) for value in df["action.act"]])
    action_executed = np.stack([_as_vec(value) for value in df["action.executed"]])
    is_intervention = np.asarray(
        [_as_scalar(value, 0.0) >= 0.5 for value in df["is_intervention"]], dtype=np.bool_
    )
    has_human_action = (
        np.asarray([_as_scalar(value, 0.0) >= 0.5 for value in df["has_human_action"]], dtype=np.bool_)
        if "has_human_action" in df.columns
        else is_intervention.copy()
    )
    episode_indices = (
        np.asarray([int(_as_scalar(value)) for value in df["episode_index"]], dtype=np.int64)
        if "episode_index" in df.columns
        else np.zeros(len(df), dtype=np.int64)
    )
    classes = build_gripper_classes(
        action_act,
        action_executed,
        is_intervention,
        has_human_action,
        episode_indices,
        hysteresis_enabled=cfg.gripper_hysteresis_enabled,
        open_threshold=cfg.gripper_open_threshold,
        close_threshold=cfg.gripper_close_threshold,
        single_threshold=cfg.gripper_single_threshold,
        act_confirm_frames=cfg.gripper_act_confirm_frames,
    )
    counts = count_gripper_events(classes, episode_indices)
    require_gripper_event_counts(counts, minimum)
    return counts


def discretize_gripper_commands(
    commands: np.ndarray,
    episode_indices: np.ndarray,
    *,
    hysteresis_enabled: bool,
    open_threshold: float,
    close_threshold: float,
    single_threshold: float,
    confirm_frames: int = 1,
) -> np.ndarray:
    """Convert continuous two-gripper commands into latched open/closed states."""
    values = np.asarray(commands, dtype=np.float32).reshape(-1, 2)
    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if len(values) != len(episodes):
        raise ValueError("Gripper command and episode lengths do not match")
    thresholds = np.asarray([open_threshold, single_threshold, close_threshold], dtype=np.float64)
    if not np.all(np.isfinite(thresholds)) or not np.all((0.0 <= thresholds) & (thresholds <= 0.8)):
        raise ValueError("Gripper thresholds must each be finite and within [0.0, 0.8]")
    if confirm_frames < 1:
        raise ValueError("Gripper confirm_frames must be at least 1")
    output = np.zeros_like(values, dtype=np.int64)
    state = np.zeros(2, dtype=np.bool_)
    pending_state = np.zeros(2, dtype=np.bool_)
    pending_count = np.zeros(2, dtype=np.int64)
    previous_episode: int | None = None
    for index, (command, episode) in enumerate(zip(values, episodes)):
        if previous_episode is None or int(episode) != previous_episode:
            threshold = close_threshold if hysteresis_enabled else single_threshold
            state = command >= threshold - GRIPPER_THRESHOLD_EPSILON
            pending_state = state.copy()
            pending_count.fill(0)
        elif hysteresis_enabled:
            requested_state = np.where(
                state,
                command > open_threshold + GRIPPER_THRESHOLD_EPSILON,
                command >= close_threshold - GRIPPER_THRESHOLD_EPSILON,
            )
        else:
            requested_state = command >= single_threshold - GRIPPER_THRESHOLD_EPSILON
        if previous_episode is not None and int(episode) == previous_episode:
            for side in range(2):
                requested = bool(requested_state[side])
                if requested == bool(state[side]):
                    pending_state[side] = state[side]
                    pending_count[side] = 0
                else:
                    if requested != bool(pending_state[side]):
                        pending_state[side] = requested
                        pending_count[side] = 1
                    else:
                        pending_count[side] += 1
                    if pending_count[side] >= confirm_frames:
                        state[side] = requested
                        pending_count[side] = 0
        output[index] = state.astype(np.int64)
        previous_episode = int(episode)
    return output


def build_gripper_classes(
    action_act: np.ndarray,
    action_executed: np.ndarray,
    is_intervention: np.ndarray,
    has_human_action: np.ndarray,
    episode_indices: np.ndarray,
    *,
    hysteresis_enabled: bool,
    open_threshold: float,
    close_threshold: float,
    single_threshold: float,
    act_confirm_frames: int = 1,
) -> np.ndarray:
    base_gripper = discretize_gripper_commands(
        np.asarray(action_act)[:, GRIPPER_INDICES],
        episode_indices,
        hysteresis_enabled=hysteresis_enabled,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
        single_threshold=single_threshold,
        confirm_frames=act_confirm_frames,
    )
    executed_gripper = discretize_gripper_commands(
        np.asarray(action_executed)[:, GRIPPER_INDICES],
        episode_indices,
        hysteresis_enabled=hysteresis_enabled,
        open_threshold=open_threshold,
        close_threshold=close_threshold,
        single_threshold=single_threshold,
        confirm_frames=1,
    )
    classes = np.zeros((len(base_gripper), 2), dtype=np.int64)
    valid_intervention = np.asarray(is_intervention, dtype=np.bool_) & np.asarray(
        has_human_action, dtype=np.bool_
    )
    differs = executed_gripper != base_gripper
    classes[valid_intervention[:, None] & differs & (executed_gripper == 0)] = 1
    classes[valid_intervention[:, None] & differs & (executed_gripper == 1)] = 2
    return classes


def count_gripper_events(classes: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    labels = np.asarray(classes, dtype=np.int64).reshape(-1, 2)
    episodes = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    if len(labels) != len(episodes):
        raise ValueError("Gripper class and episode lengths do not match")
    counts = np.zeros((2, 3), dtype=np.int64)
    previous_class = np.zeros(2, dtype=np.int64)
    previous_episode: int | None = None
    for values, episode in zip(labels, episodes):
        episode_changed = previous_episode is None or int(episode) != previous_episode
        for side in range(2):
            value = int(values[side])
            if value != 0 and (episode_changed or value != int(previous_class[side])):
                counts[side, value] += 1
        previous_class = values
        previous_episode = int(episode)
    return counts


def require_gripper_event_counts(counts: np.ndarray, minimum: int) -> None:
    if minimum < 1:
        raise ValueError("Minimum gripper event count must be at least 1")
    missing = []
    for side, side_name in enumerate(("left", "right")):
        for class_index in (1, 2):
            if counts[side, class_index] < minimum:
                missing.append(
                    f"{side_name} {GRIPPER_CLASS_NAMES[class_index]}="
                    f"{counts[side, class_index]} < {minimum}"
                )
    if missing:
        raise ValueError("Insufficient independent gripper correction events: " + ", ".join(missing))


class _ResidualDataBase:
    REQUIRED_COLUMNS = (
        "observation.state",
        "action.act",
        "action.human",
        "action.rl_delta",
        "action.executed",
        "is_intervention",
    )

    def __init__(self, cfg: ResidualDatasetConfig) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root).expanduser().resolve()
        self.df = read_lerobot_parquets(self.root)
        missing = [key for key in self.REQUIRED_COLUMNS if key not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset is missing residual fields: {missing}")
        if cfg.visual_cache is None:
            raise ValueError("A three-camera ACT visual cache is required.")
        self.visual_cache = (
            cfg.visual_cache
            if isinstance(cfg.visual_cache, VisualCache)
            else load_visual_feature_cache(cfg.visual_cache, expected_total_frames=len(self.df))
        )
        if len(self.visual_cache.features) != len(self.df):
            raise ValueError("Visual feature cache and parquet frame count do not match")
        self.visual_features = self.visual_cache.features
        self.states = np.stack([_as_vec(value) for value in self.df["observation.state"]]).astype(np.float32)
        self.action_act = np.stack([_as_vec(value) for value in self.df["action.act"]]).astype(np.float32)
        self.action_executed = np.stack([_as_vec(value) for value in self.df["action.executed"]]).astype(np.float32)
        self.is_intervention = np.asarray(
            [_as_scalar(value, 0.0) >= 0.5 for value in self.df["is_intervention"]], dtype=np.bool_
        )
        if "has_human_action" in self.df.columns:
            self.has_human_action = np.asarray(
                [_as_scalar(value, 0.0) >= 0.5 for value in self.df["has_human_action"]], dtype=np.bool_
            )
        else:
            self.has_human_action = self.is_intervention.copy()
        self.residual_limits = np.asarray(cfg.residual_limits, dtype=np.float32).reshape(len(ARM_INDICES))
        if np.any(np.abs(self.residual_limits) < 1e-8):
            raise ValueError("All residual limits must be non-zero")
        if cfg.residual_lambda <= 0:
            raise ValueError("residual_lambda must be positive")
        self.obs_dim = int(self.visual_features.shape[1]) + 32
        self.action_dim = len(ARM_INDICES)
        self.dataset_action_dim = len(JOINT_NAMES)
        if "episode_index" in self.df.columns:
            self.episode_indices = np.asarray(
                [int(_as_scalar(value)) for value in self.df["episode_index"]], dtype=np.int64
            )
        else:
            self.episode_indices = np.zeros(len(self.df), dtype=np.int64)
        self.gripper_classes = build_gripper_classes(
            self.action_act,
            self.action_executed,
            self.is_intervention,
            self.has_human_action,
            self.episode_indices,
            hysteresis_enabled=cfg.gripper_hysteresis_enabled,
            open_threshold=cfg.gripper_open_threshold,
            close_threshold=cfg.gripper_close_threshold,
            single_threshold=cfg.gripper_single_threshold,
            act_confirm_frames=cfg.gripper_act_confirm_frames,
        )
        self.observation_stats = cfg.observation_stats
        if self.observation_stats is not None and self.observation_stats.mean.size != self.obs_dim:
            raise ValueError(
                f"Observation stats dim={self.observation_stats.mean.size}, expected obs_dim={self.obs_dim}"
            )

    @property
    def act_fingerprint(self) -> str:
        return self.visual_cache.act_fingerprint

    @property
    def image_keys(self) -> list[str]:
        return list(self.visual_cache.metadata["image_keys"])

    @property
    def visual_feature_dim(self) -> int:
        return int(self.visual_cache.feature_dim)

    def _raw_obs_at(self, index: int) -> np.ndarray:
        feature = np.asarray(self.visual_features[index], dtype=np.float32)
        return np.concatenate([feature, self.states[index], self.action_act[index]], axis=0).astype(np.float32)

    def obs_at(self, index: int) -> np.ndarray:
        raw = self._raw_obs_at(index)
        return self.observation_stats.normalize(raw) if self.observation_stats is not None else raw

    def fit_observation_stats(self, indices: Iterable[int] | None = None) -> ObservationStats:
        selected = np.asarray(list(indices) if indices is not None else np.arange(len(self.df)), dtype=np.int64)
        if selected.size == 0:
            raise ValueError("Cannot fit observation statistics on zero frames")
        total = np.zeros(self.obs_dim, dtype=np.float64)
        total_sq = np.zeros(self.obs_dim, dtype=np.float64)
        count = 0
        for start in range(0, selected.size, 4096):
            ids = selected[start : start + 4096]
            visual = np.asarray(self.visual_features[ids], dtype=np.float32)
            block = np.concatenate([visual, self.states[ids], self.action_act[ids]], axis=1).astype(np.float64)
            total += block.sum(axis=0)
            total_sq += np.square(block).sum(axis=0)
            count += block.shape[0]
        mean = total / count
        variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
        stats = ObservationStats(
            mean=mean.astype(np.float32),
            std=np.sqrt(variance).astype(np.float32),
            clip=float(self.cfg.normalization_clip),
        )
        stats.std = np.maximum(stats.std, 1e-6)
        self.observation_stats = stats
        return stats

    def _intervention_target(self, index: int) -> np.ndarray:
        human = _as_vec(self.df.iloc[index]["action.human"])
        delta_joint = (human - self.action_act[index]) / float(self.cfg.residual_lambda)
        normalized = delta_joint[ARM_INDICES] / np.abs(self.residual_limits)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _recorded_residual_target(self, index: int) -> np.ndarray:
        if self.is_intervention[index] and self.has_human_action[index]:
            return self._intervention_target(index)
        delta_joint = _as_vec(self.df.iloc[index]["action.rl_delta"])
        normalized = delta_joint[ARM_INDICES] / np.abs(self.residual_limits)
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def gripper_event_counts(self, indices: Iterable[int] | None = None) -> np.ndarray:
        selected = np.ones(len(self.df), dtype=np.bool_)
        if indices is not None:
            selected[:] = False
            selected[np.asarray(list(indices), dtype=np.int64)] = True
        counts = np.zeros((2, 3), dtype=np.int64)
        previous_class = np.zeros(2, dtype=np.int64)
        previous_episode: int | None = None
        for index, (classes, episode) in enumerate(zip(self.gripper_classes, self.episode_indices)):
            episode_changed = previous_episode is None or int(episode) != previous_episode
            if selected[index]:
                for side in range(2):
                    value = int(classes[side])
                    if value != 0 and (episode_changed or value != int(previous_class[side])):
                        counts[side, value] += 1
            previous_class = classes
            previous_episode = int(episode)
        return counts

    def validate_gripper_event_counts(self, minimum: int) -> np.ndarray:
        counts = self.gripper_event_counts()
        require_gripper_event_counts(counts, minimum)
        return counts


class ResidualBCDataset(_ResidualDataBase, Dataset):
    """Supervised residual policy dataset.

    Intervention frames imitate ``action.human - action.act``.  Non-intervention
    frames use an explicit zero residual target, preventing a policy that always
    applies a correction.
    """

    def __init__(self, cfg: ResidualDatasetConfig, *, intervention_loss_weight: float = 3.0) -> None:
        super().__init__(cfg)
        if intervention_loss_weight <= 0:
            raise ValueError("intervention_loss_weight must be positive")
        self.intervention_loss_weight = float(intervention_loss_weight)
        valid_intervention = self.is_intervention & self.has_human_action
        self.intervention_indices = np.flatnonzero(valid_intervention).astype(np.int64).tolist()
        self.non_intervention_indices = np.flatnonzero(~valid_intervention).astype(np.int64).tolist()
        self.indices = list(range(len(self.df)))
        if not self.intervention_indices:
            raise ValueError("Residual BC needs at least one intervention frame with action.human")
        if not self.non_intervention_indices:
            raise ValueError("Residual BC needs at least one non-intervention frame for zero-residual supervision")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_index = self.indices[index]
        intervention = bool(self.is_intervention[frame_index] and self.has_human_action[frame_index])
        target = self._intervention_target(frame_index) if intervention else np.zeros(self.action_dim, dtype=np.float32)
        return {
            "obs": torch.as_tensor(self.obs_at(frame_index), dtype=torch.float32),
            "action": torch.as_tensor(target, dtype=torch.float32),
            "sample_weight": torch.as_tensor(
                [self.intervention_loss_weight if intervention else 1.0], dtype=torch.float32
            ),
            "is_intervention": torch.as_tensor([1.0 if intervention else 0.0], dtype=torch.float32),
            "gripper_class": torch.as_tensor(self.gripper_classes[frame_index], dtype=torch.long),
        }


class ResidualTransitionDataset(_ResidualDataBase, Dataset):
    """Build normalized visual transitions for offline residual SAC/CQL."""

    def __init__(self, cfg: ResidualDatasetConfig) -> None:
        super().__init__(cfg)
        self.rewards = read_reward_column(self.df)
        self.done = compute_done(self.df)
        self.indices: list[int] = []
        self.next_indices: list[int] = []
        episode = None
        if "episode_index" in self.df.columns:
            episode = np.asarray([int(_as_scalar(v)) for v in self.df["episode_index"]], dtype=np.int64)
        for index in range(len(self.df)):
            if cfg.use_only_interventions and not (self.is_intervention[index] and self.has_human_action[index]):
                continue

            # BW reward annotation stores terminal reward/done on the last frame.
            # Keep that frame as a terminal transition and use a self-loop next_obs;
            # the Bellman target masks next_q because done=1. Dropping it would lose
            # the most important success/failure reward.
            if self.done[index] >= 0.5:
                next_index = index
            else:
                next_index = index + 1
                if next_index >= len(self.df):
                    raise ValueError(
                        "The final dataset frame is not marked done; annotate episode termination before RL training."
                    )
                if episode is not None and episode[next_index] != episode[index]:
                    raise ValueError(
                        f"Frame {index} crosses an episode boundary without done=1; fix the dataset metadata."
                    )
            self.indices.append(index)
            self.next_indices.append(next_index)
        if not self.indices:
            raise ValueError("No usable transitions found. Check episode lengths, done and intervention fields.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frame_index = self.indices[index]
        next_index = self.next_indices[index]
        return {
            "obs": torch.as_tensor(self.obs_at(frame_index), dtype=torch.float32),
            "action": torch.as_tensor(self._recorded_residual_target(frame_index), dtype=torch.float32),
            "reward": torch.as_tensor([self.rewards[frame_index]], dtype=torch.float32),
            "next_obs": torch.as_tensor(self.obs_at(next_index), dtype=torch.float32),
            "done": torch.as_tensor([self.done[frame_index]], dtype=torch.float32),
        }
