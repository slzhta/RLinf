"""Regression tests for success-only rewards and episode-boundary bootstrapping."""

from inspect import unwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.maniskill.tasks.digital_twin.pick_and_place import (
    PickAndPlaceDigitalTwinEnv,
)
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg
from rlinf.workers.env.env_worker import EnvWorker

CONFIG = Path(__file__).resolve().parents[2] / "examples/embodiment/config"


def bootstrap(mode, output, values, reward_model_output=None, auto_reset=True):
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {"bootstrap_type": mode, "gamma": 0.9},
                "env": {"train": {"auto_reset": auto_reset}},
            }
        ),
        env_reward_weight=0.5,
        reward_weight=2.0,
    )
    return unwrap(EnvWorker.compute_bootstrap_rewards)(
        worker, output, values, reward_model_output
    )


def boundary_output():
    # Running, timeout, true terminal, and simultaneous terminal+timeout.
    terminations = torch.tensor([[False], [False], [True], [True]])
    truncations = torch.tensor([[False], [True], [False], [True]])
    return EnvOutput(
        obs={},
        rewards=torch.tensor([[0.0], [0.0], [1.0], [1.0]]),
        terminations=terminations,
        truncations=truncations,
        dones=terminations | truncations,
    )


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("none", [0, 0, 1, 1]),
        ("standard", [0, 1.8, 1, 4.6]),
        ("always", [0, 1.8, 3.7, 4.6]),
    ],
)
def test_bootstrap_modes_follow_slz_truncation_mask(mode, expected):
    output = boundary_output()
    original = output.rewards.clone()
    result = bootstrap(mode, output, torch.arange(1, 5).float()[:, None])
    torch.testing.assert_close(result[:, 0], torch.tensor(expected).float())
    torch.testing.assert_close(output.rewards, original)


def test_none_never_reads_bootstrap_values_and_keeps_reward_mixing():
    output = boundary_output()
    values = torch.full((4, 1), float("nan"))
    result = bootstrap("none", output, values, torch.ones(4, 1))
    torch.testing.assert_close(result, output.rewards * 0.5 + 2.0)


def test_invalid_bootstrap_rejected_when_bootstrap_is_evaluated():
    with pytest.raises(ValueError, match="bootstrap_type"):
        bootstrap("typo", boundary_output(), torch.ones(4, 1))


@pytest.mark.parametrize("auto_reset,values", [(False, torch.ones(4, 1)), (True, None)])
def test_no_bootstrap_without_values_or_auto_reset(auto_reset, values):
    output = boundary_output()
    torch.testing.assert_close(
        bootstrap("standard", output, values, auto_reset=auto_reset), output.rewards
    )


def test_none_terminal_returns_do_not_leak_reset_values():
    output = boundary_output()
    rewards = bootstrap("none", output, torch.full((4, 1), 100.0))
    _, returns = compute_gae_advantages_and_returns(
        rewards=rewards.T,
        values=torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]]),
        dones=torch.cat([torch.zeros(1, 4, dtype=torch.bool), output.dones.T]),
        gamma=0.9,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    # A live rollout boundary keeps ordinary GAE bootstrap; ended episodes do not.
    torch.testing.assert_close(returns, torch.tensor([[9.0, 0.0, 1.0, 1.0]]))


def test_sparse_reward_ignores_all_milestones_and_failures():
    info = {
        "success": torch.tensor([False, False, False, False, True]),
        "fail": torch.tensor([False, False, False, True, False]),
        "is_grasped": torch.ones(5, dtype=torch.bool),
        "is_obj_lifted": torch.ones(5, dtype=torch.bool),
        "is_obj_placed": torch.ones(5, dtype=torch.bool),
        "cube_to_goal_dist": torch.tensor([0.5, 0.2, 0.01, 1.0, 0.0]),
    }
    cfg = residual_config()
    # Exercise the callback actually selected by the merged configuration.
    env = object.__new__(PickAndPlaceDigitalTwinEnv)
    env.task_alignment = dict(cfg.env.train.init_params.task_alignment)
    for key in ("grasp", "lift", "place", "success"):
        setattr(env, f"_sparse_{key}_rewarded", torch.zeros(5, dtype=torch.bool))
    env._sparse_drop_penalized = torch.zeros(5, dtype=torch.bool)
    reward = env.compute_dense_reward(None, None, info)
    torch.testing.assert_close(env.compute_dense_reward(None, None, info), torch.zeros(5))
    torch.testing.assert_close(reward, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))


def residual_config():
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.1"):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_a4")
    OmegaConf.resolve(cfg)
    return cfg


def test_a4_resolved_configuration_uses_only_success_rewards():
    cfg = residual_config()
    validate_residual_cfg(cfg)
    assert cfg.algorithm.bootstrap_type == "none"
    for split in ("train", "eval"):
        assert cfg.env[split].init_params.reward_mode == "dense"
        assert cfg.env[split].init_params.sim_config.control_freq == 10


@pytest.mark.parametrize("split", ["train", "eval"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("init_params.reward_mode", "sparse"),
        ("init_params.reward_mode", "normalized_dense"),
        ("init_params.task_alignment.use_dense_reward", True),
        ("init_params.task_alignment.sparse_grasp_reward", 0.1),
        ("init_params.task_alignment.sparse_lift_reward", 0.2),
        ("init_params.task_alignment.sparse_place_reward", 0.3),
        ("init_params.task_alignment.sparse_drop_penalty", -0.2),
        ("reward_mode", "default"),
        ("use_rel_reward", True),
        ("ignore_terminations", True),
    ],
)
def test_dense_and_shaping_overrides_are_rejected(split, key, value):
    cfg = residual_config()
    OmegaConf.update(cfg, f"env.{split}.{key}", value)
    with pytest.raises(ValueError, match="success-only"):
        validate_residual_cfg(cfg)


def test_external_reward_model_is_rejected():
    cfg = residual_config()
    cfg.reward.use_reward_model = True
    with pytest.raises(ValueError, match="forbid reward models"):
        validate_residual_cfg(cfg)
