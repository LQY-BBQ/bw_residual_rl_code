from __future__ import annotations

import numpy as np
import torch

from bw_datasets.residual_transition_dataset import (
    build_gripper_classes,
    count_gripper_events,
    discretize_gripper_commands,
)
from policies.residual_bc_policy import DeterministicResidualActor
from policies.residual_sac_policy import Critic, SquashedGaussianActor
from train_residual_sac import initialize_actor_from_bc


def test_bc_and_sac_hybrid_shapes_and_frozen_gripper() -> None:
    bc = DeterministicResidualActor(12, 14, [16, 16])
    obs = torch.randn(4, 12)
    arm, logits = bc(obs)
    assert arm.shape == (4, 14)
    assert logits.shape == (4, 2, 3)

    sac = SquashedGaussianActor(12, 14, [16, 16])
    config = {
        "obs_dim": 12,
        "action_dim": 14,
        "hidden_dims": [16, 16],
        "act_fingerprint": "fingerprint",
        "obs_mode": "act_visual_state_act",
        "residual_lambda": 0.2,
        "residual_limits": [0.03] * 14,
    }
    initialize_actor_from_bc(
        sac,
        bc_config=config,
        bc_state=bc.state_dict(),
        obs_dim=12,
        action_dim=14,
        hidden_dims=[16, 16],
        act_fingerprint_value="fingerprint",
        residual_lambda=0.2,
        residual_limits=np.full(14, 0.03, dtype=np.float32),
    )
    critic = Critic(12, 14, [16])
    assert critic(obs, sac.act(obs)).shape == (4, 1)

    frozen = DeterministicResidualActor(12, 14, [16, 16])
    frozen.load_state_dict(bc.state_dict())
    frozen.requires_grad_(False)
    before = frozen(obs)[1].detach().clone()
    optimizer = torch.optim.Adam(sac.parameters(), lr=1e-3)
    optimizer.zero_grad()
    sac.act(obs).square().mean().backward()
    optimizer.step()
    after = frozen(obs)[1].detach()
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_episode_gripper_discretization_modes() -> None:
    commands = np.asarray([[0.1, 0.5], [0.3, 0.3], [0.4, 0.2]], dtype=np.float32)
    episodes = np.zeros(3, dtype=np.int64)
    hysteresis = discretize_gripper_commands(
        commands,
        episodes,
        hysteresis_enabled=True,
        open_threshold=0.2,
        close_threshold=0.4,
        single_threshold=0.3,
    )
    np.testing.assert_array_equal(hysteresis, [[0, 1], [0, 1], [1, 0]])
    single = discretize_gripper_commands(
        commands,
        episodes,
        hysteresis_enabled=False,
        open_threshold=0.2,
        close_threshold=0.4,
        single_threshold=0.3,
    )
    np.testing.assert_array_equal(single, [[0, 1], [1, 1], [1, 0]])


def test_gripper_discretization_confirms_act_transitions() -> None:
    commands = np.asarray(
        [
            [0.8, 0.0],
            [0.49, 0.41],
            [0.49, 0.39],
            [0.49, 0.41],
            [0.39, 0.41],
            [0.39, 0.41],
        ],
        dtype=np.float32,
    )
    states = discretize_gripper_commands(
        commands,
        np.zeros(len(commands), dtype=np.int64),
        hysteresis_enabled=True,
        open_threshold=0.50,
        close_threshold=0.40,
        single_threshold=0.45,
        confirm_frames=3,
    )
    np.testing.assert_array_equal(
        states,
        [[1, 0], [1, 0], [1, 0], [0, 0], [0, 0], [0, 1]],
    )


def test_gripper_labels_ignore_non_intervention_zero_human_actions() -> None:
    act = np.zeros((5, 16), dtype=np.float32)
    executed = np.zeros((5, 16), dtype=np.float32)
    act[:, [7, 15]] = [0.0, 0.8]
    executed[:, [7, 15]] = [0.8, 0.0]
    intervention = np.asarray([False, True, True, False, True])
    classes = build_gripper_classes(
        act,
        executed,
        intervention,
        intervention,
        np.zeros(5, dtype=np.int64),
        hysteresis_enabled=False,
        open_threshold=0.2,
        close_threshold=0.4,
        single_threshold=0.3,
    )
    np.testing.assert_array_equal(classes[0], [0, 0])
    np.testing.assert_array_equal(classes[1], [2, 1])
    np.testing.assert_array_equal(classes[3], [0, 0])
    counts = count_gripper_events(classes, np.zeros(5, dtype=np.int64))
    np.testing.assert_array_equal(counts[:, 1:], [[0, 2], [2, 0]])
