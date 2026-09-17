# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check the shared-feature PnP co-training deployment without workers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg


@pytest.fixture(params=[16, 32])
def cfg(request: pytest.FixtureRequest) -> DictConfig:
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        return compose(
            config_name=f"co_rl_pick_and_place_openpi_residual_shared_t1_n{request.param}_ue4"
        )


def test_shared_pnp_residual_contract(cfg: DictConfig) -> None:
    validate_residual_cfg(cfg)
    model = cfg.actor.model
    assert model.gripper_architecture == "shared"
    assert model.gripper_share_features
    assert model.binary_action_temperature == 1.0
    assert model.encoder_config.freeze_backbone
    assert model.encoder_input_size == 128
    assert model.initial_logstd == -0.5
    assert model.gripper == {}
    assert model.shared_feature_checkpoint.endswith("best_features.pt")
    assert model.gripper_checkpoint.endswith("best_head.pt")
    assert cfg.runner.ckpt_path is None and cfg.runner.resume_dir is None


def test_pnp_paired_sampling_and_real_inputs(cfg: DictConfig) -> None:
    assert cfg.cluster.num_nodes == 3
    assert cfg.algorithm.co_training_rollout_routing_mode == "paired"
    buffer = cfg.algorithm.co_training_domain_buffer
    assert buffer.mode == "latest"
    assert (buffer.real_ratio_min, buffer.real_ratio_target, buffer.real_ratio_max) == (
        0.05,
        0.1,
        0.15,
    )
    samples = buffer.sample_rollout_length * cfg.env.train.max_steps_per_rollout_epoch
    assert samples == 4320 and samples % cfg.actor.global_batch_size == 0
    assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0
    assert cfg.algorithm.update_epoch == 4
    expected_envs = 16 if "_n16_" in cfg.runner.logger.experiment_name else 32
    assert cfg.env.train.total_num_envs == expected_envs
    assert cfg.env.eval.total_num_envs == expected_envs
    real = cfg.env.train.co_training_env_cfg
    assert real.total_num_envs == 1
    assert real.keyboard_reward_wrapper == "pnp_human"
    assert list(real.state_keys) == [
        "arm_joint_position",
        "tcp_pose",
        "gripper_open_state",
    ]
    assert real.main_image_key == "wrist_2"
    assert real.human_feedback_cfg.wait_for_reset_ready
    assert real.video_cfg.save_video and cfg.env.train.video_cfg.save_video
    assert cfg.runner.save_interval == 5
    assert cfg.runner.val_check_interval == -1


def test_pnp_periodic_eval_stays_disabled(cfg: DictConfig) -> None:
    cfg.runner.val_check_interval = 5
    with pytest.raises(ValueError, match="without evaluation"):
        validate_residual_cfg(cfg)
