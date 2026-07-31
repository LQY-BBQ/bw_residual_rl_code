"""Deterministic residual behavior-cloning actor."""
from __future__ import annotations

import torch
from torch import nn

from .residual_sac_policy import MLP


class DeterministicResidualActor(nn.Module):
    """Hybrid actor with continuous arm residuals and categorical grippers."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.trunk = MLP(obs_dim, hidden_dims, hidden_dims[-1])
        if action_dim != 14:
            raise ValueError(f"Hybrid residual BC requires 14 arm actions, got {action_dim}")
        self.arm_mu = nn.Linear(hidden_dims[-1], action_dim)
        self.gripper_logits = nn.Linear(hidden_dims[-1], 6)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        arm = torch.tanh(self.arm_mu(hidden))
        gripper = self.gripper_logits(hidden).reshape(-1, 2, 3)
        return arm, gripper

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs)
