"""Train visual residual BC with frozen ACT image features."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import time
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset

from bw_datasets.residual_transition_dataset import (
    ResidualBCDataset,
    ResidualDatasetConfig,
    preflight_gripper_event_counts,
)
from policies.act_shared_encoder import act_policy_fingerprint
from policies.residual_bc_policy import DeterministicResidualActor
from visual_cache import build_visual_feature_cache, default_cache_dir

JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_yaw_joint", "left_shoulder_roll_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "left_gripper_joint",
    "right_shoulder_pitch_joint", "right_shoulder_yaw_joint", "right_shoulder_roll_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint", "right_gripper_joint",
]


class BalancedInterventionBatchSampler(Sampler[list[int]]):
    """Draw an exact intervention/non-intervention ratio with replacement."""

    def __init__(
        self,
        intervention_indices: list[int],
        non_intervention_indices: list[int],
        *,
        batch_size: int,
        intervention_ratio: float,
        batches_per_epoch: int,
        seed: int,
    ) -> None:
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if not 0.0 < intervention_ratio < 1.0:
            raise ValueError("intervention_ratio must be between 0 and 1")
        self.intervention_indices = list(intervention_indices)
        self.non_intervention_indices = list(non_intervention_indices)
        self.batch_size = int(batch_size)
        self.n_intervention = max(1, min(batch_size - 1, int(round(batch_size * intervention_ratio))))
        self.n_non_intervention = batch_size - self.n_intervention
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches_per_epoch):
            batch = [rng.choice(self.intervention_indices) for _ in range(self.n_intervention)]
            batch += [rng.choice(self.non_intervention_indices) for _ in range(self.n_non_intervention)]
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACT-visual residual behavior cloning.")
    parser.add_argument("--dataset.root", dest="dataset_root", type=Path, required=True)
    parser.add_argument("--dataset.repo_id", dest="repo_id", default=None)
    parser.add_argument("--act-policy-path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--intervention-ratio", type=float, default=0.5)
    parser.add_argument("--intervention-loss-weight", type=float, default=3.0)
    parser.add_argument("--residual-lambda", type=float, default=0.2)
    parser.add_argument("--residual-limit-default", type=float, default=0.03)
    parser.add_argument("--residual-limit-gripper", type=float, default=None)
    parser.add_argument("--gripper-loss-weight", type=float, default=1.0)
    parser.add_argument("--gripper-min-events", type=int, default=20)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    hysteresis = parser.add_mutually_exclusive_group()
    hysteresis.add_argument("--gripper-hysteresis", dest="gripper_hysteresis", action="store_true")
    hysteresis.add_argument("--no-gripper-hysteresis", dest="gripper_hysteresis", action="store_false")
    parser.set_defaults(gripper_hysteresis=True)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.20)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.40)
    parser.add_argument("--gripper-single-threshold", type=float, default=0.30)
    parser.add_argument("--normalization-clip", type=float, default=10.0)
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


def validate_gripper_args(args: argparse.Namespace) -> None:
    thresholds = (
        float(args.gripper_open_threshold),
        float(args.gripper_single_threshold),
        float(args.gripper_close_threshold),
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("Gripper thresholds must be finite")
    if not 0.0 <= thresholds[0] < thresholds[1] < thresholds[2] <= 0.8:
        raise ValueError(
            "Gripper thresholds must satisfy 0.0 <= open < single < close <= 0.8"
        )
    if args.gripper_min_events < 20:
        raise ValueError("--gripper-min-events cannot be lower than the required minimum of 20")


def split_episode_indices(dataset: ResidualBCDataset, *, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    episodes = np.unique(dataset.episode_indices)
    if episodes.size < 2:
        raise ValueError("Residual BC validation requires at least two episodes")
    validation_count = max(1, min(episodes.size - 1, int(round(episodes.size * validation_ratio))))
    rng = np.random.default_rng(seed)
    for _ in range(2000):
        shuffled = rng.permutation(episodes)
        validation_episodes = set(shuffled[:validation_count].tolist())
        train_indices = [i for i, episode in enumerate(dataset.episode_indices) if episode not in validation_episodes]
        validation_indices = [i for i, episode in enumerate(dataset.episode_indices) if episode in validation_episodes]
        train_classes = dataset.gripper_classes[train_indices]
        validation_classes = dataset.gripper_classes[validation_indices]
        required = all(
            np.any(train_classes[:, side] == cls) and np.any(validation_classes[:, side] == cls)
            for side in range(2)
            for cls in (1, 2)
        )
        if required:
            return train_indices, validation_indices
    raise ValueError("Could not create an episode-level split containing every FORCE class in train and validation")


def gripper_class_weights(dataset: ResidualBCDataset, indices: list[int]) -> torch.Tensor:
    labels = dataset.gripper_classes[indices]
    weights = np.ones((2, 3), dtype=np.float32)
    for side in range(2):
        counts = np.bincount(labels[:, side], minlength=3).astype(np.float32)
        if np.any(counts <= 0):
            raise ValueError(f"Training split has missing gripper class on side={side}: {counts.tolist()}")
        weights[side] = np.minimum(np.sqrt(counts[0] / counts), 10.0)
    return torch.as_tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    actor: DeterministicResidualActor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    actor.eval()
    arm_error = 0.0
    left_correct = 0
    right_correct = 0
    count = 0
    for batch in loader:
        obs = batch["obs"].to(device)
        target = batch["action"].to(device)
        gripper_target = batch["gripper_class"].to(device)
        arm, logits = actor(obs)
        batch_count = int(obs.shape[0])
        arm_error += float((arm - target).abs().mean(dim=-1).sum().cpu())
        prediction = logits.argmax(dim=-1)
        left_correct += int((prediction[:, 0] == gripper_target[:, 0]).sum().cpu())
        right_correct += int((prediction[:, 1] == gripper_target[:, 1]).sum().cpu())
        count += batch_count
    actor.train()
    if count == 0:
        raise ValueError("Validation loader is empty")
    return arm_error / count, left_correct / count, right_correct / count


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


def save_checkpoint(path: Path, actor: DeterministicResidualActor, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "config": config}, path)
    (path.parent / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    validate_gripper_args(args)
    set_seed(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    limits = residual_limits(args.residual_limit_default, args.residual_limit_gripper)
    if args.residual_limit_gripper is not None:
        print("WARNING: --residual-limit-gripper is deprecated and ignored; grippers are categorical.")
    dataset_config = ResidualDatasetConfig(
        root=args.dataset_root,
        residual_limits=limits,
        residual_lambda=args.residual_lambda,
        normalization_clip=args.normalization_clip,
        gripper_hysteresis_enabled=args.gripper_hysteresis,
        gripper_open_threshold=args.gripper_open_threshold,
        gripper_close_threshold=args.gripper_close_threshold,
        gripper_single_threshold=args.gripper_single_threshold,
    )
    event_counts = preflight_gripper_event_counts(dataset_config, args.gripper_min_events)
    print(
        "Gripper correction events: "
        f"left open/close={event_counts[0, 1]}/{event_counts[0, 2]}, "
        f"right open/close={event_counts[1, 1]}/{event_counts[1, 2]}"
    )
    visual_cache = _prepare_visual_cache(args)
    dataset_config.visual_cache = visual_cache
    dataset = ResidualBCDataset(
        dataset_config,
        intervention_loss_weight=args.intervention_loss_weight,
    )
    train_indices, validation_indices = split_episode_indices(
        dataset, validation_ratio=args.validation_ratio, seed=args.seed
    )
    obs_stats = dataset.fit_observation_stats(train_indices)
    intervention_set = set(dataset.intervention_indices)
    non_intervention_set = set(dataset.non_intervention_indices)
    train_interventions = [index for index in train_indices if index in intervention_set]
    train_non_interventions = [index for index in train_indices if index in non_intervention_set]
    if not train_interventions or not train_non_interventions:
        raise ValueError(
            "Episode split must leave both intervention and non-intervention frames in the training set"
        )
    batches_per_epoch = max(1, math.ceil(len(train_indices) / args.batch_size))
    batch_sampler = BalancedInterventionBatchSampler(
        train_interventions,
        train_non_interventions,
        batch_size=args.batch_size,
        intervention_ratio=args.intervention_ratio,
        batches_per_epoch=batches_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.batch_size, shuffle=False)
    data_iter = iter(loader)
    class_weights = gripper_class_weights(dataset, train_indices).to(device)

    actor = DeterministicResidualActor(dataset.obs_dim, dataset.action_dim, args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "format_version": 4,
        "policy_type": "residual_bc",
        "obs_mode": "act_visual_state_act",
        "obs_dim": dataset.obs_dim,
        "visual_feature_dim": dataset.visual_feature_dim,
        "action_dim": dataset.action_dim,
        "dataset_action_dim": dataset.dataset_action_dim,
        "hidden_dims": list(args.hidden_dims),
        "observation_stats": obs_stats.to_dict(),
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
        "intervention_ratio": float(args.intervention_ratio),
        "intervention_loss_weight": float(args.intervention_loss_weight),
        "loss": "smooth_l1",
        "gripper_class_names": ["KEEP_BASE", "FORCE_OPEN", "FORCE_CLOSE"],
        "gripper_loss_weight": float(args.gripper_loss_weight),
        "gripper_class_weights": class_weights.cpu().tolist(),
        "gripper_event_counts": event_counts.tolist(),
        "gripper_control": {
            "open_value": 0.0,
            "close_value": 0.8,
            "residual_confidence_threshold": 0.70,
            "residual_confirm_frames": 3,
            "min_hold_s": 0.30,
            "hysteresis_enabled": bool(args.gripper_hysteresis),
            "open_threshold": float(args.gripper_open_threshold),
            "close_threshold": float(args.gripper_close_threshold),
            "single_threshold": float(args.gripper_single_threshold),
        },
    }
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_path.write_text(
        "step,loss,arm_loss,gripper_loss,intervention_mae,non_intervention_mae,"
        "left_gripper_accuracy,right_gripper_accuracy,pred_abs_mean,elapsed_s\n",
        encoding="utf-8",
    )
    validation_path = args.output_dir / "validation_metrics.csv"
    validation_path.write_text(
        "step,arm_mae,left_gripper_accuracy,right_gripper_accuracy\n", encoding="utf-8"
    )
    t0 = time.time()
    actor.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        obs = batch["obs"].to(device)
        target = batch["action"].to(device)
        weights = batch["sample_weight"].to(device)
        intervention = batch["is_intervention"].to(device).reshape(-1) >= 0.5
        gripper_target = batch["gripper_class"].to(device)
        prediction, gripper_logits = actor(obs)
        per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1, keepdim=True)
        arm_loss = (per_sample * weights).mean()
        left_gripper_loss = F.cross_entropy(gripper_logits[:, 0], gripper_target[:, 0], weight=class_weights[0])
        right_gripper_loss = F.cross_entropy(gripper_logits[:, 1], gripper_target[:, 1], weight=class_weights[1])
        gripper_loss = 0.5 * (left_gripper_loss + right_gripper_loss)
        loss = arm_loss + args.gripper_loss_weight * gripper_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_freq == 0:
            with torch.no_grad():
                absolute_error = (prediction - target).abs().mean(dim=-1)
                intervention_mae = absolute_error[intervention].mean() if intervention.any() else torch.tensor(0.0, device=device)
                normal_mask = ~intervention
                non_intervention_mae = absolute_error[normal_mask].mean() if normal_mask.any() else torch.tensor(0.0, device=device)
                pred_abs = prediction.abs().mean()
                class_prediction = gripper_logits.argmax(dim=-1)
                left_accuracy = (class_prediction[:, 0] == gripper_target[:, 0]).float().mean()
                right_accuracy = (class_prediction[:, 1] == gripper_target[:, 1]).float().mean()
            elapsed = time.time() - t0
            line = (
                f"{step},{float(loss.detach().cpu()):.7f},{float(arm_loss.detach().cpu()):.7f},"
                f"{float(gripper_loss.detach().cpu()):.7f},{float(intervention_mae.cpu()):.7f},"
                f"{float(non_intervention_mae.cpu()):.7f},{float(left_accuracy.cpu()):.7f},"
                f"{float(right_accuracy.cpu()):.7f},{float(pred_abs.cpu()):.7f},{elapsed:.2f}\n"
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
            print(
                f"step={step} loss={float(loss):.5f} intervention_mae={float(intervention_mae):.5f} "
                f"zero_mae={float(non_intervention_mae):.5f} "
                f"gripper_acc={float(left_accuracy):.3f}/{float(right_accuracy):.3f} frames={len(dataset)}"
            )
        if step % args.save_freq == 0:
            validation_metrics = evaluate(actor, validation_loader, device)
            with validation_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"{step},{validation_metrics[0]:.7f},{validation_metrics[1]:.7f},"
                    f"{validation_metrics[2]:.7f}\n"
                )
            save_checkpoint(args.output_dir / "checkpoints" / f"step_{step:06d}" / "residual_bc.pt", actor, config)
            save_checkpoint(args.output_dir / "checkpoints" / "last" / "residual_bc.pt", actor, config)

    actor.eval()
    final_path = args.output_dir / "checkpoints" / "last" / "residual_bc.pt"
    save_checkpoint(final_path, actor, config)
    print(f"Saved final residual BC checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
