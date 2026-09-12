# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the single-node simulation contract without allocating workers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg


@pytest.fixture
def simulation_cfg():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        return compose(
            config_name="sim_rl_pick_and_place_residual_scalar_gripper_n32_ue4"
        )


def test_single_node_simulation_validates(simulation_cfg):
    validate_residual_cfg(simulation_cfg)
    assert simulation_cfg.actor.model.gripper.output_mode == "scalar"
    assert simulation_cfg.actor.model.gripper_initial_logstd == -1.0
    assert simulation_cfg.env.train.total_num_envs == 32
    assert simulation_cfg.actor.model.gripper_architecture == "independent"


def test_shared_config_requires_matching_frontend(simulation_cfg):
    simulation_cfg.actor.model.gripper_architecture = "shared"
    simulation_cfg.actor.model.gripper = {}
    with pytest.raises(ValueError, match="shared_feature_checkpoint"):
        validate_residual_cfg(simulation_cfg)
    simulation_cfg.actor.model.shared_feature_checkpoint = "/test/shared_features.pt"
    validate_residual_cfg(simulation_cfg)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("cluster.num_nodes", 3, "1 nodes"),
        ("runner.val_check_interval", 50, "without evaluation"),
        ("env.eval.env_type", "realworld", "ManiSkill"),
        ("env.train.residual.enabled", True, "env-side"),
        ("env.train.include_states_in_obs", False, "states"),
        ("env.train.init_params.task_alignment.use_dense_reward", True, "success-only"),
        ("env.train.init_params.controller_alignment.close_command", 0.0, "convention"),
    ],
)
def test_simulation_rejects_invalid_contract(simulation_cfg, key, value, match):
    OmegaConf.update(simulation_cfg, key, value, force_add=True)
    with pytest.raises(ValueError, match=match):
        validate_residual_cfg(simulation_cfg)
