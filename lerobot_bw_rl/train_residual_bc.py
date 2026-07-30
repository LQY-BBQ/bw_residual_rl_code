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
from torch.utils.data import DataLoader, Sampler

from bw_datasets.residual_transition_dataset import ResidualBCDataset, ResidualDatasetConfig
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
    parser.add_argument("--residual-limit-gripper", type=float, default=0.03)
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


def residual_limits(default: float, gripper: float) -> np.ndarray:
    values = np.full(len(JOINT_NAMES), float(default), dtype=np.float32)
    for index, name in enumerate(JOINT_NAMES):
        if name.endswith("gripper_joint"):
            values[index] = float(gripper)
    return values


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
    set_seed(args.seed)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    limits = residual_limits(args.residual_limit_default, args.residual_limit_gripper)
    visual_cache = _prepare_visual_cache(args)
    dataset = ResidualBCDataset(
        ResidualDatasetConfig(
            root=args.dataset_root,
            residual_limits=limits,
            residual_lambda=args.residual_lambda,
            visual_cache=visual_cache,
            normalization_clip=args.normalization_clip,
        ),
        intervention_loss_weight=args.intervention_loss_weight,
    )
    obs_stats = dataset.fit_observation_stats()
    batches_per_epoch = max(1, math.ceil(len(dataset) / args.batch_size))
    batch_sampler = BalancedInterventionBatchSampler(
        dataset.intervention_indices,
        dataset.non_intervention_indices,
        batch_size=args.batch_size,
        intervention_ratio=args.intervention_ratio,
        batches_per_epoch=batches_per_epoch,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers)
    data_iter = iter(loader)

    actor = DeterministicResidualActor(dataset.obs_dim, dataset.action_dim, args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = {
        "format_version": 3,
        "policy_type": "residual_bc",
        "obs_mode": "act_visual_state_act",
        "obs_dim": dataset.obs_dim,
        "visual_feature_dim": dataset.visual_feature_dim,
        "action_dim": dataset.action_dim,
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
    }
    metrics_path = args.output_dir / "train_metrics.csv"
    metrics_path.write_text(
        "step,loss,intervention_mae,non_intervention_mae,pred_abs_mean,elapsed_s\n", encoding="utf-8"
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
        prediction = actor(obs)
        per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1, keepdim=True)
        loss = (per_sample * weights).mean()
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
            elapsed = time.time() - t0
            line = (
                f"{step},{float(loss.detach().cpu()):.7f},{float(intervention_mae.cpu()):.7f},"
                f"{float(non_intervention_mae.cpu()):.7f},{float(pred_abs.cpu()):.7f},{elapsed:.2f}\n"
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
            print(
                f"step={step} loss={float(loss):.5f} intervention_mae={float(intervention_mae):.5f} "
                f"zero_mae={float(non_intervention_mae):.5f} frames={len(dataset)}"
            )
        if step % args.save_freq == 0:
            save_checkpoint(args.output_dir / "checkpoints" / f"step_{step:06d}" / "residual_bc.pt", actor, config)
            save_checkpoint(args.output_dir / "checkpoints" / "last" / "residual_bc.pt", actor, config)

    actor.eval()
    final_path = args.output_dir / "checkpoints" / "last" / "residual_bc.pt"
    save_checkpoint(final_path, actor, config)
    print(f"Saved final residual BC checkpoint: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
