from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import bw_datasets.residual_transition_dataset as residual_dataset_module
from bw_datasets.residual_transition_dataset import (
    ResidualDatasetConfig,
    build_gripper_classes,
    count_gripper_events,
    discretize_gripper_commands,
    preflight_gripper_event_counts,
)
from policies.residual_bc_policy import DeterministicResidualActor
from policies.residual_sac_policy import Critic, SquashedGaussianActor
from train_residual_bc import parse_args, validate_gripper_args, warn_if_relaxed_gripper_minimum
from train_residual_sac import initialize_actor_from_bc


def _gripper_args(minimum: int) -> SimpleNamespace:
    return SimpleNamespace(
        gripper_open_threshold=0.50,
        gripper_single_threshold=0.45,
        gripper_close_threshold=0.40,
        gripper_min_events=minimum,
        gripper_act_confirm_frames=3,
    )


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


@pytest.mark.parametrize("minimum", [1, 10, 20])
def test_validate_gripper_args_allows_positive_minimum(minimum: int) -> None:
    validate_gripper_args(_gripper_args(minimum))


def test_parse_args_keeps_recommended_gripper_minimum_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_residual_bc.py",
            "--dataset.root",
            "/tmp/dataset",
            "--act-policy-path",
            "/tmp/act",
            "--output_dir",
            "/tmp/output",
        ],
    )
    assert parse_args().gripper_min_events == 20


def test_validate_gripper_args_rejects_non_positive_minimum() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        validate_gripper_args(_gripper_args(0))


def test_relaxed_gripper_minimum_prints_warning(capsys: pytest.CaptureFixture[str]) -> None:
    warn_if_relaxed_gripper_minimum(10)
    assert "WARNING" in capsys.readouterr().err

    warn_if_relaxed_gripper_minimum(20)
    assert capsys.readouterr().err == ""


def test_preflight_gripper_event_counts_uses_requested_minimum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    action_act = np.zeros((2, 16), dtype=np.float32)
    action_executed = np.zeros((2, 16), dtype=np.float32)
    action_act[0, [7, 15]] = 0.8
    action_executed[1, [7, 15]] = 0.8
    frame_table = pd.DataFrame(
        {
            "action.act": list(action_act),
            "action.executed": list(action_executed),
            "is_intervention": [1.0, 1.0],
            "has_human_action": [1.0, 1.0],
            "episode_index": [0, 1],
        }
    )
    monkeypatch.setattr(
        residual_dataset_module,
        "read_lerobot_parquets",
        lambda _root: frame_table,
    )
    config = ResidualDatasetConfig(
        root=tmp_path,
        residual_limits=np.full(14, 0.2, dtype=np.float32),
    )

    counts = preflight_gripper_event_counts(config, minimum=1)
    np.testing.assert_array_equal(counts[:, 1:], [[1, 1], [1, 1]])
    with pytest.raises(ValueError, match="Insufficient independent gripper correction events"):
        preflight_gripper_event_counts(config, minimum=2)
