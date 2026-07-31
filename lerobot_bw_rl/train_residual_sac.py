"""Offline visual residual SAC/CQL training on BW ACT-correction data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bw_datasets.residual_transition_dataset import (
    ObservationStats,
    ResidualDatasetConfig,
    ResidualTransitionDataset,
)
from policies.act_shared_encoder import act_policy_fingerprint
from policies.residual_bc_policy import DeterministicResidualActor
from policies.residual_sac_policy import Critic, SACBatch, SquashedGaussianActor
from visual_cache import build_visual_feature_cache, default_cache_dir

JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_gripper_joint",
    "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_gripper_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-ACT visual residual SAC/CQL.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo_id", dest="repo_id", default=None)
    parser.add_argument("--act-policy-path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--init-from-bc", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--alpha_lr", type=float, default=3e-4)
    parser.add_argument("--init_alpha", type=float, default=0.2)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--bc-loss-weight", type=float, default=0.0)
    parser.add_argument("--cql-alpha", type=float, default=0.0)
    parser.add_argument("--demo-sample-ratio", type=float, default=0.0, help="Reserved for future RLPD buffers.")
    parser.add_argument("--residual-lambda", type=float, default=0.2)
    parser.add_argument("--residual-limit-default", type=float, default=0.03)
    parser.add_argument("--residual-limit-gripper", type=float, default=None)
    parser.add_argument("--normalization-clip", type=float, default=10.0)
    parser.add_argument("--use-only-interventions", action="store_true")
    parser.add_argument("--visual-feature-mode", choices=["cache", "online"], default="cache")
    parser.add_argument("--visual-cache-dir", type=Path, default=None)
    parser.add_argument("--visual-cache-batch-size", type=int, default=16)
    parser.add_argument("--visual-cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--visual-cache-use-amp", action="store_true")
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--rebuild-visual-cache", action="store_true")
    parser.add_argument("--save_freq", type=int, default=2000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def residual_limits(default: float, gripper: float | None) -> np.ndarray:  # noqa: ARG001
    return np.full(14, float(default), dtype=np.float32)


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> SACBatch:
    return SACBatch(
        obs=batch["obs"].to(device),
        action=batch["action"].to(device),
        reward=batch["reward"].to(device),
        next_obs=batch["next_obs"].to(device),
        done=batch["done"].to(device),
    )


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.data.mul_(1.0 - tau).add_(source_parameter.data, alpha=tau)


def _resolve_checkpoint(path: str | Path, names: tuple[str, ...]) -> Path:
    value = Path(path).expanduser().resolve()
    if value.is_dir():
        candidates = [value / name for name in names]
        candidates += [value / "checkpoints" / "last" / name for name in names]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if value.exists():
        return value
    raise FileNotFoundError(f"Checkpoint not found: {value}")


def load_bc_initialization(path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], Path]:
    checkpoint_path = _resolve_checkpoint(path, ("residual_bc.pt", "checkpoint.pt", "last.pt"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if int(config.get("format_version", -1)) != 4:
        raise ValueError("--init-from-bc requires a format-v4 hybrid residual BC checkpoint")
    if config.get("policy_type") != "residual_bc":
        raise ValueError(f"--init-from-bc requires a residual_bc checkpoint, got {config.get('policy_type')}")
    if int(config.get("action_dim", -1)) != 14 or int(config.get("dataset_action_dim", -1)) != 16:
        raise ValueError("--init-from-bc requires action_dim=14 and dataset_action_dim=16")
    if config.get("gripper_class_names") != ["KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE"]:
        raise ValueError("--init-from-bc checkpoint has an invalid gripper class mapping")
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
        raise ValueError("--init-from-bc checkpoint is missing gripper control metadata")
    state_dict = checkpoint.get("actor")
    if not isinstance(state_dict, dict):
        raise ValueError("Residual BC checkpoint is missing actor state_dict")
    return config, state_dict, checkpoint_path


def initialize_actor_from_bc(
    actor: SquashedGaussianActor,
    *,
    bc_config: dict[str, Any],
    bc_state: dict[str, torch.Tensor],
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    act_fingerprint_value: str,
    residual_lambda: float,
    residual_limits: np.ndarray,
    camera_contract_metadata: dict[str, Any] | None = None,
) -> None:
    expected = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_dims": list(hidden_dims),
        "act_fingerprint": act_fingerprint_value,
        "obs_mode": "act_visual_state_act",
    }
    for key, value in expected.items():
        if bc_config.get(key) != value:
            raise ValueError(
                f"Residual BC initialization mismatch for {key}: checkpoint={bc_config.get(key)!r}, expected={value!r}"
            )
    for key, value in (camera_contract_metadata or {}).items():
        if bc_config.get(key) != value:
            raise ValueError(
                f"Residual BC camera contract mismatch for {key}: "
                f"checkpoint={bc_config[key]!r}, expected={value!r}"
            )
    bc_lambda = float(bc_config.get("residual_lambda", float("nan")))
    if not np.isclose(bc_lambda, residual_lambda):
        raise ValueError(
            "Residual BC and SAC must use the same residual_lambda: "
            f"BC={bc_lambda}, SAC={residual_lambda}"
        )
    bc_limits = np.asarray(bc_config.get("residual_limits", []), dtype=np.float32).reshape(-1)
    if bc_limits.shape != residual_limits.shape or not np.allclose(bc_limits, residual_limits):
        raise ValueError(
            "Residual BC and SAC must use identical residual_limits because the actor output is normalized by them."
        )
    trunk_state = {
        key.removeprefix("trunk."): value
        for key, value in bc_state.items()
        if key.startswith("trunk.")
    }
    actor.trunk.load_state_dict(trunk_state)
    try:
        actor.mu.weight.data.copy_(bc_state["arm_mu.weight"])
        actor.mu.bias.data.copy_(bc_state["arm_mu.bias"])
    except KeyError as exc:
        raise ValueError(f"Residual BC checkpoint is missing hybrid arm head: {exc}") from exc
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(-2.0)


def save_checkpoint(
    path: Path,
    *,
    actor: SquashedGaussianActor,
    gripper_actor: DeterministicResidualActor,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "gripper_actor": gripper_actor.state_dict(),
            "config": config,
        },
        path,
    )
    (path.parent / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _prepare_visual_cache(args: argparse.Namespace):
    fingerprint = act_policy_fingerprint(args.act_policy_path)
    if args.visual_feature_mode == "online":
        cache_dir = args.output_dir.expanduser() / ".online_visual_features"
        overwrite = True
    else:
        cache_dir = args.visual_cache_dir or default_cache_dir(args.dataset_root, fingerprint)
        overwrite = bool(args.rebuild_visual_cache)
    return build_visual_feature_cache(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        act_policy_path=args.act_policy_path,
        cache_dir=cache_dir,
        device=args.device,
        use_amp=args.visual_cache_use_amp,
        batch_size=args.visual_cache_batch_size,
        dtype=args.visual_cache_dtype,
        overwrite=overwrite,
        video_backend=args.video_backend,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    limits = residual_limits(args.residual_limit_default, args.residual_limit_gripper)
    if args.residual_limit_gripper is not None:
        print("WARNING: --residual-limit-gripper is deprecated and ignored; SAC trains arm actions only.")
    visual_cache = _prepare_visual_cache(args)

    bc_config, bc_state, bc_path = load_bc_initialization(args.init_from_bc)
    observation_stats = ObservationStats.from_dict(bc_config["observation_stats"])
    gripper_control = dict(bc_config.get("gripper_control", {}))

    dataset = ResidualTransitionDataset(
        ResidualDatasetConfig(
            root=args.dataset_root,
            residual_limits=limits,
            residual_lambda=args.residual_lambda,
            visual_cache=visual_cache,
            observation_stats=observation_stats,
            normalization_clip=args.normalization_clip,
            use_only_interventions=args.use_only_interventions,
            gripper_hysteresis_enabled=bool(gripper_control.get("hysteresis_enabled", True)),
            gripper_open_threshold=float(gripper_control.get("open_threshold", 0.50)),
            gripper_close_threshold=float(gripper_control.get("close_threshold", 0.40)),
            gripper_single_threshold=float(gripper_control.get("single_threshold", 0.45)),
            # Old format-v4 BC checkpoints predate ACT confirmation and imply one frame.
            gripper_act_confirm_frames=int(gripper_control.get("act_confirm_frames", 1)),
        )
    )
    if dataset.observation_stats is None:
        observation_stats = dataset.fit_observation_stats()
    else:
        observation_stats = dataset.observation_stats

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(dataset) >= args.batch_size,
        num_workers=args.num_workers,
    )
    data_iter = iter(loader)
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim
    actor = SquashedGaussianActor(obs_dim, action_dim, args.hidden_dims).to(device)
    initialize_actor_from_bc(
        actor,
        bc_config=bc_config,
        bc_state=bc_state,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        act_fingerprint_value=dataset.act_fingerprint,
        residual_lambda=args.residual_lambda,
        residual_limits=limits,
        camera_contract_metadata={
            key: visual_cache.metadata.get(key)
            for key in (
                "dataset_fps",
                "source_image_shapes",
                "policy_image_shapes",
                "camera_contract_version",
                "image_transform",
            )
        },
    )
    gripper_actor = DeterministicResidualActor(obs_dim, action_dim, args.hidden_dims).to(device)
    gripper_actor.load_state_dict(bc_state)
    gripper_actor.eval()
    for parameter in gripper_actor.parameters():
        parameter.requires_grad_(False)
    print(f"Initialized SAC arm actor and frozen gripper actor from: {bc_path}")
    q1 = Critic(obs_dim, action_dim, args.hidden_dims).to(device)
    q2 = Critic(obs_dim, action_dim, args.hidden_dims).to(device)
    q1_target = Critic(obs_dim, action_dim, args.hidden_dims).to(device)
    q2_target = Critic(obs_dim, action_dim, args.hidden_dims).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.critic_lr)
    log_alpha = torch.tensor(np.log(args.init_alpha), dtype=torch.float32, device=device, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
    target_entropy = float(args.target_entropy if args.target_entropy is not None else -action_dim)

    config: dict[str, Any] = {
        "format_version": 4,
        "policy_type": "residual_rl",
        "algorithm": "offline_sac_cql",
        "obs_mode": "act_visual_state_act",
        "obs_dim": obs_dim,
        "visual_feature_dim": dataset.visual_feature_dim,
        "action_dim": action_dim,
        "dataset_action_dim": dataset.dataset_action_dim,
        "hidden_dims": list(args.hidden_dims),
        "observation_stats": observation_stats.to_dict(),
        "residual_limits": limits.tolist(),
        "residual_lambda": float(args.residual_lambda),
        "action_is_normalized": True,
        "act_policy_path": str(args.act_policy_path.expanduser()),
        "act_fingerprint": dataset.act_fingerprint,
        "image_keys": dataset.image_keys,
        "visual_feature_definition": visual_cache.metadata["feature_definition"],
        "dataset_fps": visual_cache.metadata.get("dataset_fps"),
        "source_image_shapes": visual_cache.metadata.get("source_image_shapes"),
        "policy_image_shapes": visual_cache.metadata.get("policy_image_shapes"),
        "camera_contract_version": visual_cache.metadata.get("camera_contract_version"),
        "image_transform": visual_cache.metadata.get("image_transform"),
        "act_parameters_frozen": True,
        "bc_loss_weight": float(args.bc_loss_weight),
        "cql_alpha": float(args.cql_alpha),
        "target_entropy": target_entropy,
        "init_from_bc": str(bc_path) if bc_path else None,
        "gripper_class_names": ["KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE"],
        "gripper_control": gripper_control,
        "gripper_policy_frozen": True,
    }
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_path.write_text(
        "step,critic_loss,actor_loss,alpha,alpha_loss,bc_loss,cql_loss,q_data,reward_mean,elapsed_s\n",
        encoding="utf-8",
    )
    t0 = time.time()
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        transition = to_device(batch, device)
        alpha = log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_log_probability = actor.sample(transition.next_obs)
            target_q = torch.min(
                q1_target(transition.next_obs, next_action),
                q2_target(transition.next_obs, next_action),
            ) - alpha * next_log_probability
            target = transition.reward + args.gamma * (1.0 - transition.done) * target_q
        q1_prediction = q1(transition.obs, transition.action)
        q2_prediction = q2(transition.obs, transition.action)
        critic_loss = F.mse_loss(q1_prediction, target) + F.mse_loss(q2_prediction, target)

        cql_loss = torch.tensor(0.0, device=device)
        if args.cql_alpha > 0:
            random_action = torch.empty_like(transition.action).uniform_(-1.0, 1.0)
            policy_action, _ = actor.sample(transition.obs)
            q1_random, q2_random = q1(transition.obs, random_action), q2(transition.obs, random_action)
            q1_policy, q2_policy = q1(transition.obs, policy_action), q2(transition.obs, policy_action)
            cql_q1 = torch.logsumexp(
                torch.cat([q1_random, q1_policy, q1_prediction], dim=1), dim=1
            ).mean() - q1_prediction.mean()
            cql_q2 = torch.logsumexp(
                torch.cat([q2_random, q2_policy, q2_prediction], dim=1), dim=1
            ).mean() - q2_prediction.mean()
            cql_loss = args.cql_alpha * (cql_q1 + cql_q2)
            critic_loss = critic_loss + cql_loss

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optimizer.step()

        new_action, log_probability = actor.sample(transition.obs)
        q_new = torch.min(q1(transition.obs, new_action), q2(transition.obs, new_action))
        actor_loss = (log_alpha.exp().detach() * log_probability - q_new).mean()
        bc_loss = torch.tensor(0.0, device=device)
        if args.bc_loss_weight > 0:
            deterministic_action = torch.tanh(actor.forward(transition.obs)[0])
            bc_loss = F.mse_loss(deterministic_action, transition.action)
            actor_loss = actor_loss + args.bc_loss_weight * bc_loss
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optimizer.step()

        alpha_loss = -(log_alpha * (log_probability.detach() + target_entropy)).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()
        soft_update(q1_target, q1, args.tau)
        soft_update(q2_target, q2, args.tau)

        if step % args.log_freq == 0:
            elapsed = time.time() - t0
            line = (
                f"{step},{float(critic_loss.detach().cpu()):.6f},{float(actor_loss.detach().cpu()):.6f},"
                f"{float(log_alpha.exp().detach().cpu()):.6f},{float(alpha_loss.detach().cpu()):.6f},"
                f"{float(bc_loss.detach().cpu()):.6f},{float(cql_loss.detach().cpu()):.6f},"
                f"{float(q1_prediction.mean().detach().cpu()):.6f},{float(transition.reward.mean().detach().cpu()):.6f},"
                f"{elapsed:.2f}\n"
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
            print(
                f"step={step} critic={float(critic_loss):.4f} actor={float(actor_loss):.4f} "
                f"alpha={float(log_alpha.exp()):.4f} transitions={len(dataset)}"
            )
        if step % args.save_freq == 0:
            save_checkpoint(
                args.output_dir / "checkpoints" / f"step_{step:06d}" / "residual_rl.pt",
                actor=actor,
                gripper_actor=gripper_actor,
                config=config,
            )
            save_checkpoint(
                args.output_dir / "checkpoints" / "last" / "residual_rl.pt",
                actor=actor,
                gripper_actor=gripper_actor,
                config=config,
            )

    final_path = args.output_dir / "checkpoints" / "last" / "residual_rl.pt"
    save_checkpoint(final_path, actor=actor, gripper_actor=gripper_actor, config=config)
    print(f"Saved final residual RL checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
