# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate the real-only PnP baseline without Ray or robot workers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def configs():
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/embodiment/config")
    ):
        return (
            compose(
                config_name="real_rl_pick_and_place_openpi_residual_shared_t1_t240_ue4"
            ),
            compose(
                config_name="co_rl_pick_and_place_openpi_residual_shared_t1_n16_ue4"
            ),
        )


def test_same_pnp_initial_policy_and_optimizer(configs):
    cfg, co = configs
    validate_residual_cfg(cfg)
    assert cfg.actor == co.actor
    assert cfg.base_model == co.base_model
    assert cfg.runner.ckpt_path is None and cfg.runner.resume_dir is None
    assert cfg.actor.model.gripper_checkpoint.endswith("best_head.pt")
    assert cfg.actor.model.shared_feature_checkpoint.endswith("best_features.pt")
    assert cfg.actor.model.encoder_config.freeze_backbone
    assert cfg.actor.model.gripper_share_features
    assert cfg.actor.model.binary_action_temperature == 1.0


def test_real_only_placement_and_sampling(configs):
    cfg, _ = configs
    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.node_groups[0].node_ranks == 0
    assert cfg.cluster.node_groups[1].node_ranks == 1
    assert cfg.cluster.component_placement.actor.node_group == "gpu"
    assert cfg.cluster.component_placement.rollout.node_group == "gpu"
    assert cfg.cluster.component_placement.env.node_group == "real"
    assert cfg.cluster.component_placement.rollout.placement == 0
    assert not cfg.algorithm.sim_real_rl_co_training
    assert "co_training_domain_buffer" not in cfg.algorithm
    steps = (
        cfg.env.train.total_num_envs
        * cfg.env.train.max_steps_per_rollout_epoch
        * cfg.algorithm.rollout_epoch
    )
    assert steps == cfg.actor.global_batch_size == 240
    assert cfg.actor.micro_batch_size == 120
    assert cfg.algorithm.update_epoch == 4
    assert cfg.rollout.pipeline_stage_num == 1


def test_real_env_matches_co_training(configs):
    cfg, co = configs
    assert cfg.env.sim_real_alignment == co.env.sim_real_alignment
    for mode in ("train", "eval"):
        real = cfg.env[mode]
        reference = co.env[mode].co_training_env_cfg
        for key in (
            "env_type",
            "init_params",
            "override_cfg",
            "state_keys",
            "main_image_key",
            "human_feedback_cfg",
            "keyboard_reward_wrapper",
            "max_episode_steps",
            "auto_reset",
            "include_states_in_obs",
        ):
            assert real[key] == reference[key], key
        assert real.total_num_envs == real.num_workers == 1
        assert "co_training_env_cfg" not in real
    assert cfg.env.train.video_cfg.save_video
    assert cfg.env.train.video_cfg.include_extra_views
    assert cfg.env.train.video_cfg.record_rollout_interval == 1
    assert cfg.runner.save_interval == 5
    assert cfg.runner.val_check_interval == -1
    assert cfg.runner.max_steps == 500


def test_periodic_eval_is_rejected(configs):
    cfg, _ = configs
    cfg.runner.val_check_interval = 5
    with pytest.raises(ValueError, match="periodic eval is disabled"):
        validate_residual_cfg(cfg)
