"""Deterministic residual behavior-cloning actor."""
from __future__ import annotations

import torch
from torch import nn

from .residual_sac_policy import MLP


class DeterministicResidualActor(nn.Module):
    """MLP actor whose tanh output is a normalized residual in [-1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.trunk = MLP(obs_dim, hidden_dims, hidden_dims[-1])
        self.mu = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mu(self.trunk(obs)))

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)
