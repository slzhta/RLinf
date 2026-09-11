# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for BC initialization and the joint Gaussian/Bernoulli PPO."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_decoupled_ppo_actor_loss
from rlinf.data.gripper_bc_dataset import GripperBCDataset
from rlinf.envs.residual import BaseActionCache
from rlinf.models.embodiment.modules.resnet_utils import ResNet10
from rlinf.models.embodiment.residual_policy.gripper_cnn import (
    GripperCNN,
    GripperCNNConfig,
)
from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
    SplitGripperConfig,
    SplitGripperPolicy,
)
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg


@pytest.fixture
def policy(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.save(ResNet10().state_dict(), tmp_path / "encoder.pt")
    cfg = SplitGripperConfig()
    cfg.update_from_dict(
        {
            "model_path": str(tmp_path),
            "encoder_config": {"ckpt_name": "encoder.pt", "dropout": 0.0},
            "use_state": True,
            "state_dim": 14,
            "image_size": [3, 32, 32],
            "image_num": 2,
            "action_dim": 7,
            "base_horizon": 3,
            "add_value_head": True,
            "add_q_head": False,
            "initial_logstd": -0.5,
            "logstd_range": [-4.0, 0.0],
            "gripper": {
                "image_num": 2,
                "image_size": 32,
                "state_dim": 14,
                "hidden_dim": 32,
            },
        }
    )
    return SplitGripperPolicy(cfg)


def observations(batch=4):
    return {
        "main_images": torch.randint(256, (batch, 32, 32, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            256, (batch, 1, 32, 32, 3), dtype=torch.uint8
        ),
        "states": torch.randn(batch, 14),
        "base_actions": torch.randn(batch, 3, 7),
        "base_action_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


def test_replayed_distribution_and_joint_ppo_update(policy):
    _, rollout = policy.predict_action_batch(observations())
    inputs = rollout["forward_inputs"]
    output = policy.default_forward(inputs)
    assert policy.actor_logstd.shape == (1, 6)
    assert output["entropy"].shape == (4, 1, 7)
    torch.testing.assert_close(
        output["logprobs"], rollout["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    new = output["logprobs"].sum(-1)
    old = rollout["prev_logprobs"].sum(-1)
    loss, _ = compute_decoupled_ppo_actor_loss(
        logprobs=new,
        old_logprobs=old,
        advantages=torch.tensor([1.0, -1.0, 0.5, -0.5]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    before_arm = policy.actor_logstd.detach().clone()
    before_gripper = policy.gripper.head[-1].weight.detach().clone()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    assert policy.actor_logstd.grad.abs().sum() > 0
    assert policy.actor_mean.weight.grad.abs().sum() > 0
    assert policy.gripper.encoder[0].weight.grad.abs().sum() > 0
    assert policy.gripper.head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before_arm, policy.actor_logstd)
    assert not torch.equal(before_gripper, policy.gripper.head[-1].weight)


def test_openpi_gripper_cannot_affect_policy_or_execution(policy):
    obs = observations()
    first, _ = policy.predict_action_batch(obs, mode="eval")
    conditioned = policy._prepare_cnn_obs(obs)["states"]
    obs["base_actions"][..., 6] = float("nan")
    torch.testing.assert_close(conditioned, policy._prepare_cnn_obs(obs)["states"])
    second, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(first, second)
    for base_gripper in (-1.0, 1.0):
        cache = BaseActionCache(2, 3, 7, torch.device("cpu"))
        cache.actions.fill_(0.2)
        cache.actions[..., 6] = base_gripper
        cache.position.zero_()
        actions = torch.zeros(2, 7)
        actions[:, 6] = torch.tensor([-1.0, 1.0])
        scale = torch.tensor([0.4] * 6 + [0.0])
        result = cache.compose(actions, scale, gripper_mode="cnn")
        torch.testing.assert_close(result[:, :6], torch.full((2, 6), 0.2))
        torch.testing.assert_close(result[:, 6], actions[:, 6])


def test_gripper_bc_checkpoint_and_joint_checkpoint_roundtrip(policy, tmp_path):
    obs = observations()
    inputs = policy._gripper_obs(policy._prepare_cnn_obs(obs))
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    optimizer = torch.optim.AdamW(policy.gripper.parameters(), lr=0.002)
    first_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        policy.gripper(inputs), target
    ).item()
    for _ in range(15):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            policy.gripper(inputs), target
        )
        loss.backward()
        optimizer.step()
    assert (
        torch.nn.functional.binary_cross_entropy_with_logits(
            policy.gripper(inputs), target
        ).item()
        < first_loss
    )
    path = tmp_path / "bc.pt"
    policy.gripper.save_bc(path)
    policy.residual_cfg.gripper_checkpoint = str(path)
    loaded = SplitGripperPolicy(policy.residual_cfg)
    torch.testing.assert_close(loaded.gripper(inputs), policy.gripper(inputs))
    loaded.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(
        loaded.predict_action_batch(obs, mode="eval")[0],
        policy.predict_action_batch(obs, mode="eval")[0],
    )
    incompatible = GripperCNN(GripperCNNConfig(image_num=1))
    with pytest.raises(ValueError, match="mismatch"):
        incompatible.load_bc(path)


def test_eval_disabled_residual_still_uses_bc_gripper(policy):
    policy.residual_cfg.enabled = False
    actions, _ = policy.predict_action_batch(observations(), mode="eval")
    assert torch.count_nonzero(actions[..., :6]) == 0
    assert torch.all(actions[..., 6].abs() == 1)


def test_dataset_alignment_labels_and_bad_encoding(tmp_path):
    path = tmp_path / "episode_000000.npz"
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    images[:, 0, 0, 0] = np.arange(4)
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = [-1, -1, 1, 1]
    np.savez(
        path,
        base_images=images,
        wrist_images=images,
        states=np.ones((4, 14), np.float32),
        actions=actions,
    )
    dataset = GripperBCDataset([path], ["base_images", "wrist_images"], 14)
    assert sorted(dataset.episode_order(3)) == list(range(4))
    for i in range(4):
        obs, label, switch = dataset[i]
        assert obs["main_images"][0, 0, 0] == i
        assert label.item() == (i >= 2)
        assert switch.item()
    with pytest.raises(ValueError, match="encoding"):
        GripperBCDataset([path], ["base_images"], 14, action_encoding="zero_one")


def test_split_config_rejects_wrong_environment_and_missing_bc():
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(
            config_name="maniskill_pick_and_place_ppo_residual_cnn_gripper_a4",
            overrides=["actor.model.gripper_checkpoint=/tmp/gripper-bc.pt"],
        )
    validate_residual_cfg(cfg)
    bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    bad.env.train.residual.gripper_mode = "residual"
    with pytest.raises(ValueError, match="gripper_mode"):
        validate_residual_cfg(bad)
    cfg.actor.model.gripper_checkpoint = None
    with pytest.raises(ValueError, match="gripper_checkpoint"):
        validate_residual_cfg(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_rollout_cpu_replay_and_joint_backward(policy):
    policy = policy.cuda()
    actions, result = policy.predict_action_batch(observations(), mode="train")
    inputs = result["forward_inputs"]
    assert inputs["main_images"].device.type == "cpu"
    assert actions.shape[-1] == 7
    assert torch.isfinite(result["prev_logprobs"]).all()
    gpu_inputs = {key: value.cuda() for key, value in inputs.items()}
    output = policy.default_forward(
        forward_inputs=gpu_inputs, compute_logprobs=True, compute_entropy=True
    )
    torch.testing.assert_close(
        output["logprobs"], result["prev_logprobs"].cuda(), rtol=1e-4, atol=1e-5
    )
    (-output["logprobs"].sum()).backward()
    assert policy.gripper.encoder[0].weight.grad.abs().sum() > 0
    assert policy.actor_logstd.grad.abs().sum() > 0


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_gripper_matches_seven_axis_cnn_distribution(policy, temperature):
    from types import SimpleNamespace

    from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy

    policy.residual_cfg.binary_action_temperature = temperature
    obs = observations()
    _, replay = policy.predict_action_batch(obs, mode="train")
    inputs = replay["forward_inputs"]
    logits = policy.gripper(policy._gripper_obs(inputs))
    means = torch.cat((torch.zeros_like(inputs["action"][:, :6]), logits), -1)
    legacy = SimpleNamespace(
        _continuous_action_indices=(),
        _binary_action_indices=(6,),
        cfg=SimpleNamespace(binary_action_temperature=temperature),
    )
    expected_logprob, expected_entropy = CNNPolicy._hybrid_action_statistics(
        legacy, means, torch.ones_like(means), inputs["action"]
    )
    actual = policy.default_forward(forward_inputs=inputs)
    torch.testing.assert_close(actual["logprobs"][:, 6], expected_logprob[:, 6])
    torch.testing.assert_close(actual["entropy"][..., 6].reshape(-1), expected_entropy[:, 6])
    torch.testing.assert_close(actual["logprobs"], replay["prev_logprobs"])
    (-actual["logprobs"].sum()).backward()
    assert policy.gripper.encoder[0].weight.grad.abs().sum() > 0
    eval_actions, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(
        eval_actions[:, 0, 6], ((logits[:, 0] >= 0).float() * 2 - 1)
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_gripper_rejects_invalid_temperature(policy, temperature):
    policy.residual_cfg.binary_action_temperature = temperature
    with pytest.raises(ValueError, match="finite and positive"):
        SplitGripperPolicy(policy.residual_cfg)
