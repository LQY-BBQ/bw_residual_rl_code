"""Runtime loader for visual residual BC and visual residual RL checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .constants import (
    BW_IMAGE_HWC_SHAPES,
    BW_IMAGE_KEYS,
    BW_IMAGE_SHAPES,
    CAMERA_CONTRACT_VERSION,
    IMAGE_TRANSFORM,
    JOINT_NAMES,
)

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(last, hidden), nn.ReLU()])
            last = hidden
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class DeterministicResidualActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.trunk = MLP(input_dim, hidden_dims, hidden_dims[-1])
        if action_dim != 14:
            raise ValueError(f"Hybrid residual BC requires 14 arm actions, got {action_dim}")
        self.arm_mu = nn.Linear(hidden_dims[-1], action_dim)
        self.gripper_logits = nn.Linear(hidden_dims[-1], 6)

    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = True,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observation)
        return torch.tanh(self.arm_mu(hidden)), self.gripper_logits(hidden).reshape(-1, 2, 3)


class GaussianResidualActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.trunk = MLP(input_dim, hidden_dims, hidden_dims[-1])
        self.mu = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observation)
        return self.mu(hidden), torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def act(self, observation: torch.Tensor, *, deterministic: bool = True) -> torch.Tensor:
        mean, log_std = self.forward(observation)
        latent = mean if deterministic else torch.distributions.Normal(mean, log_std.exp()).rsample()
        return torch.tanh(latent)


@dataclass(slots=True)
class ResidualPolicyBundle:
    actor: nn.Module
    gripper_actor: DeterministicResidualActor
    policy_type: str
    device: torch.device
    input_dim: int
    visual_feature_dim: int
    action_dim: int
    residual_limits: np.ndarray
    residual_lambda: float
    observation_mean: np.ndarray
    observation_std: np.ndarray
    normalization_clip: float
    act_fingerprint: str
    image_keys: tuple[str, ...]
    source_image_shapes: dict[str, tuple[int, int, int]]
    policy_image_shapes: dict[str, tuple[int, int]]
    camera_contract_version: int
    image_transform: str
    dataset_fps: float
    checkpoint_path: Path
    config: dict[str, Any]
    gripper_control: dict[str, Any]


@dataclass(slots=True)
class ResidualInference:
    arm_delta_normalized: np.ndarray
    gripper_classes: np.ndarray
    gripper_confidences: np.ndarray


def _resolve_checkpoint(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        names = ("residual_bc.pt", "residual_rl.pt", "residual_sac.pt", "checkpoint.pt", "last.pt")
        candidates = [value / name for name in names]
        candidates += [value / "checkpoints" / "last" / name for name in names]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if value.exists():
        return value
    raise FileNotFoundError(f"Residual policy checkpoint not found: {value}")


def load_residual_policy(path: str | Path, *, device: str | torch.device = "cuda") -> ResidualPolicyBundle:
    checkpoint_path = _resolve_checkpoint(path)
    torch_device = torch.device(device if not isinstance(device, torch.device) else device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    config: dict[str, Any] = dict(checkpoint.get("config", {}))
    if int(config.get("format_version", -1)) != 4:
        raise ValueError("Residual checkpoint is not format-v4 hybrid arm/gripper checkpoint; retrain it.")
    policy_type = str(config.get("policy_type", "")).lower()
    if policy_type not in {"residual_bc", "residual_rl"}:
        raise ValueError(f"Unsupported residual checkpoint policy_type={policy_type!r}")
    input_dim = int(config.get("obs_dim", 0))
    action_dim = int(config.get("action_dim", len(JOINT_NAMES)))
    hidden_dims = [int(value) for value in config.get("hidden_dims", [256, 256])]
    if input_dim <= 0:
        raise ValueError("Residual checkpoint is missing a valid obs_dim")
    if action_dim != 14 or int(config.get("dataset_action_dim", -1)) != 16:
        raise ValueError("Format-v4 residual checkpoints require action_dim=14 and dataset_action_dim=16")
    if config.get("gripper_class_names") != ["KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE"]:
        raise ValueError("Format-v4 residual checkpoint has an invalid gripper class mapping")
    gripper_control = config.get("gripper_control")
    required_gripper_metadata = {
        "open_value",
        "close_value",
        "residual_confidence_threshold",
        "residual_confirm_frames",
        "min_hold_s",
        "hysteresis_enabled",
        "open_threshold",
        "single_threshold",
        "close_threshold",
    }
    if not isinstance(gripper_control, dict) or not required_gripper_metadata.issubset(gripper_control):
        raise ValueError("Format-v4 residual checkpoint is missing gripper control metadata")
    gripper_actor = DeterministicResidualActor(input_dim, action_dim, hidden_dims)
    if policy_type == "residual_bc":
        actor: nn.Module = gripper_actor
    else:
        actor = GaussianResidualActor(input_dim, action_dim, hidden_dims)
    state_dict = checkpoint.get("actor", checkpoint.get("actor_state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError("Residual checkpoint is missing actor state_dict")
    actor.load_state_dict(state_dict)
    actor.to(torch_device).eval()
    if policy_type == "residual_rl":
        gripper_state = checkpoint.get("gripper_actor")
        if not isinstance(gripper_state, dict):
            raise ValueError("Residual RL checkpoint is missing frozen gripper_actor state_dict")
        gripper_actor.load_state_dict(gripper_state)
    gripper_actor.to(torch_device).eval()
    for parameter in gripper_actor.parameters():
        parameter.requires_grad_(False)

    stats = config.get("observation_stats")
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError("Visual residual checkpoint is missing observation_stats")
    observation_mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
    observation_std = np.maximum(np.asarray(stats["std"], dtype=np.float32).reshape(-1), 1e-6)
    if observation_mean.size != input_dim or observation_std.size != input_dim:
        raise ValueError("Residual checkpoint observation_stats dimension mismatch")
    image_keys = tuple(str(key) for key in config.get("image_keys", []))
    visual_feature_dim = int(config.get("visual_feature_dim", input_dim - 32))
    if visual_feature_dim <= 0 or input_dim != visual_feature_dim + 32:
        raise ValueError("Residual checkpoint must use [ACT visual feature, 16-D state, 16-D ACT action]")
    source_image_shapes = {
        str(key): tuple(int(value) for value in shape)
        for key, shape in (config.get("source_image_shapes") or {}).items()
    }
    policy_image_shapes = {
        str(key): tuple(int(value) for value in shape)
        for key, shape in (config.get("policy_image_shapes") or {}).items()
    }
    if image_keys != BW_IMAGE_KEYS:
        raise ValueError("Residual checkpoint camera order does not match the third-generation BW contract")
    if source_image_shapes != BW_IMAGE_HWC_SHAPES:
        raise ValueError("Residual checkpoint source image shapes do not match the third-generation BW contract")
    if policy_image_shapes != BW_IMAGE_SHAPES:
        raise ValueError("Residual checkpoint ACT image shapes do not match the third-generation BW contract")
    camera_contract_version = int(config.get("camera_contract_version", -1))
    if camera_contract_version != CAMERA_CONTRACT_VERSION:
        raise ValueError("Residual checkpoint camera contract version is stale; retrain it")
    image_transform = str(config.get("image_transform", ""))
    if image_transform != IMAGE_TRANSFORM:
        raise ValueError("Residual checkpoint must use exact camera shapes without image resizing")
    dataset_fps = float(config.get("dataset_fps", 0.0))
    if not np.isclose(dataset_fps, 30.0, rtol=0.0, atol=1e-6):
        raise ValueError("Residual checkpoint must be trained from a 30 FPS dataset")
    residual_limits = np.asarray(config.get("residual_limits", []), dtype=np.float32).reshape(-1)
    if residual_limits.size != action_dim or not np.all(np.isfinite(residual_limits)):
        raise ValueError("Format-v4 residual checkpoint must contain 14 finite residual limits")
    if np.any(residual_limits <= 0):
        raise ValueError("Format-v4 residual checkpoint residual limits must be positive")
    return ResidualPolicyBundle(
        actor=actor,
        gripper_actor=gripper_actor,
        policy_type=policy_type,
        device=torch_device,
        input_dim=input_dim,
        visual_feature_dim=visual_feature_dim,
        action_dim=action_dim,
        residual_limits=residual_limits,
        residual_lambda=float(config.get("residual_lambda", 0.2)),
        observation_mean=observation_mean,
        observation_std=observation_std,
        normalization_clip=float(stats.get("clip", 10.0)),
        act_fingerprint=str(config.get("act_fingerprint", "")),
        image_keys=image_keys,
        source_image_shapes=source_image_shapes,
        policy_image_shapes=policy_image_shapes,
        camera_contract_version=camera_contract_version,
        image_transform=image_transform,
        dataset_fps=dataset_fps,
        checkpoint_path=checkpoint_path,
        config=config,
        gripper_control=dict(gripper_control),
    )


def build_residual_runtime_obs(
    *,
    observation_state: np.ndarray,
    action_act: np.ndarray,
    act_feature: np.ndarray,
) -> np.ndarray:
    feature = np.asarray(act_feature, dtype=np.float32).reshape(-1)
    state = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    base_action = np.asarray(action_act, dtype=np.float32).reshape(-1)
    if state.size != 16 or base_action.size != 16:
        raise ValueError(f"Residual runtime requires 16-D state/action, got {state.size}/{base_action.size}")
    return np.concatenate([feature, state, base_action], axis=0).astype(np.float32)


def infer_residual_delta(
    bundle: ResidualPolicyBundle,
    residual_obs: np.ndarray,
    *,
    deterministic: bool = True,
) -> np.ndarray:
    raw = np.asarray(residual_obs, dtype=np.float32).reshape(-1)
    if raw.size != bundle.input_dim:
        raise ValueError(f"Residual policy expected obs_dim={bundle.input_dim}, got {raw.size}")
    normalized = (raw - bundle.observation_mean) / bundle.observation_std
    if bundle.normalization_clip > 0:
        normalized = np.clip(normalized, -bundle.normalization_clip, bundle.normalization_clip)
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=bundle.device).reshape(1, -1)
    with torch.inference_mode():
        action = bundle.actor.act(tensor, deterministic=deterministic)
    if isinstance(action, tuple):
        action = action[0]
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def infer_residual_action(
    bundle: ResidualPolicyBundle,
    residual_obs: np.ndarray,
    *,
    deterministic: bool = True,
) -> ResidualInference:
    raw = np.asarray(residual_obs, dtype=np.float32).reshape(-1)
    if raw.size != bundle.input_dim:
        raise ValueError(f"Residual policy expected obs_dim={bundle.input_dim}, got {raw.size}")
    normalized = (raw - bundle.observation_mean) / bundle.observation_std
    if bundle.normalization_clip > 0:
        normalized = np.clip(normalized, -bundle.normalization_clip, bundle.normalization_clip)
    tensor = torch.as_tensor(normalized, dtype=torch.float32, device=bundle.device).reshape(1, -1)
    with torch.inference_mode():
        if bundle.policy_type == "residual_bc":
            arm, logits = bundle.gripper_actor.act(tensor, deterministic=True)
        else:
            arm = bundle.actor.act(tensor, deterministic=deterministic)
            _, logits = bundle.gripper_actor.act(tensor, deterministic=True)
        probabilities = torch.softmax(logits, dim=-1)
        confidences, classes = probabilities.max(dim=-1)
    return ResidualInference(
        arm_delta_normalized=arm.detach().cpu().numpy().reshape(-1).astype(np.float32),
        gripper_classes=classes.detach().cpu().numpy().reshape(2).astype(np.int64),
        gripper_confidences=confidences.detach().cpu().numpy().reshape(2).astype(np.float32),
    )
