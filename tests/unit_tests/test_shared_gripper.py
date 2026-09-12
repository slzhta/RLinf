# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Shared-feature contracts without OpenPI, Ray workers or environments."""

from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.residual_policy.shared_gripper_policy import (
    SharedGripperConfig,
    SharedGripperPolicy,
)


@pytest.fixture
def policy(monkeypatch):
    from rlinf.models.embodiment.cnn_policy import cnn_policy

    class TinyEncoder(nn.Module):
        def __init__(self, sample_x, out_dim, encoder_cfg):
            super().__init__()
            self.out_dim = out_dim
            self.freeze_backbone = encoder_cfg.get("freeze_backbone", True)
            self.resnet_backbone = nn.Conv2d(3, 4, 1)
            self.projection = nn.Linear(4, out_dim)
            self.resnet_backbone.requires_grad_(not self.freeze_backbone)

        def forward(self, x):
            assert x.shape[-2:] == (128, 128)
            x = self.resnet_backbone(x)
            if self.freeze_backbone:
                x = x.detach()
            return self.projection(x.mean((-1, -2)))

    monkeypatch.setattr(cnn_policy, "ResNetEncoder", TinyEncoder)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = SharedGripperConfig(
        image_size=[3, 128, 128],
        image_num=2,
        state_dim=14,
        action_dim=7,
        base_horizon=3,
        add_value_head=True,
        encoder_config={"dropout": 0.0},
    )
    return SharedGripperPolicy(cfg)


def observation():
    return {
        "main_images": torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (4, 1, 32, 32, 3), dtype=torch.uint8
        ),
        "states": torch.randn(4, 14),
        "base_actions": torch.randn(4, 3, 7),
        "base_action_mask": torch.ones(4, 3, dtype=torch.bool),
    }


def distribution_outputs(policy, obs):
    prepared = policy.preprocess_env_obs(policy._prepare_cnn_obs(obs))
    return policy._actor_forward_from_processed_tensors(
        prepared["main_images"], prepared["states"], prepared["extra_view_images"]
    )


def test_zero_arm_mean_binary_gripper_and_replay(policy):
    obs = observation()
    deterministic, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(deterministic[..., :6], torch.zeros(4, 1, 6))
    assert (deterministic[..., 6].abs() == 1).all()
    actions, result = policy.predict_action_batch(obs, mode="train")
    torch.testing.assert_close(actions[:, 0], result["forward_inputs"]["action"])
    replay = policy.default_forward(result["forward_inputs"])
    torch.testing.assert_close(
        replay["logprobs"], result["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    assert replay["entropy"].shape == (4, 1, 7)
    assert replay["values"].shape[0] == 4
    assert all(p.requires_grad for p in policy.encoders.parameters())
    expected = (policy.gripper_bc_logits(obs) >= 0).float() * 2 - 1
    torch.testing.assert_close(deterministic[:, 0, 6:7], expected)
    obs["base_actions"].zero_()
    assert result["forward_inputs"]["states"][:, :21].abs().sum() > 0


def test_plan_conditions_arm_not_gripper_and_ignores_base_gripper(policy):
    with torch.no_grad():
        policy.actor_mean.weight.normal_(std=0.1)
    obs = observation()
    original = distribution_outputs(policy, obs)
    changed = {**obs, "base_actions": obs["base_actions"] + 1}
    other = distribution_outputs(policy, changed)
    assert not torch.allclose(original[2][:, :6], other[2][:, :6])
    torch.testing.assert_close(original[2][:, 6], other[2][:, 6])
    gripper_only = obs["base_actions"].clone()
    gripper_only[..., 6] += 100
    same = distribution_outputs(policy, {**obs, "base_actions": gripper_only})
    torch.testing.assert_close(original[2], same[2])
    masked = {**obs, "base_action_mask": torch.zeros(4, 3, dtype=torch.bool)}
    masked_changed = {**masked, "base_actions": changed["base_actions"]}
    torch.testing.assert_close(
        distribution_outputs(policy, masked)[2],
        distribution_outputs(policy, masked_changed)[2],
    )


def test_ppo_updates_shared_features_and_both_heads(policy):
    _, result = policy.predict_action_batch(observation(), mode="train")
    replay = policy.default_forward(result["forward_inputs"])
    ratio = (replay["logprobs"].sum(-1) - result["prev_logprobs"].sum(-1)).exp()
    advantage = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = -(ratio * advantage).mean() + 0.5 * replay["values"].square().mean()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    parameters = [
        policy.actor_mean.weight,
        policy.gripper_head[-1].weight,
        policy.encoders[0].resnet_backbone.weight,
        policy.state_proj[0].weight,
        policy.action_proj[0].weight,
        next(policy.value_head.parameters()),
    ]
    before = [p.detach().clone() for p in parameters]
    loss.backward()
    for p in parameters:
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert p.grad.abs().sum() > 0
    optimizer.step()
    assert all(not torch.equal(old, p) for old, p in zip(before, parameters))


def test_head_only_bc_matching_frontend_and_unfreeze(policy, tmp_path):
    obs = observation()
    feature_path, head_path = tmp_path / "features.pt", tmp_path / "head.pt"
    policy.save_shared_features(feature_path)
    policy.set_gripper_bc_mode()
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    assert all(
        name.startswith("gripper_head.")
        for name, p in policy.named_parameters()
        if p.requires_grad
    )
    optimizer = torch.optim.AdamW(policy.gripper_head.parameters(), lr=1e-3)
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    F.binary_cross_entropy_with_logits(policy.gripper_bc_logits(obs), target).backward()
    optimizer.step()
    assert all(
        torch.equal(v, policy.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("gripper_head.")
    )
    assert not torch.equal(
        before["gripper_head.2.weight"], policy.gripper_head[-1].weight
    )
    policy.save_gripper_bc(head_path)
    cfg = deepcopy(policy.residual_cfg)
    cfg.shared_feature_checkpoint = str(feature_path)
    cfg.gripper_checkpoint = str(head_path)
    restored = SharedGripperPolicy(cfg)
    torch.testing.assert_close(
        policy.gripper_bc_logits(obs), restored.gripper_bc_logits(obs)
    )
    assert all(p.requires_grad for p in restored.parameters())
    policy.set_rl_mode()
    assert all(p.requires_grad for p in policy.parameters())
    assert not any(e.freeze_backbone for e in policy.encoders)


def test_reject_wrong_frontend_and_legacy_head(policy, tmp_path):
    policy.save_shared_features(tmp_path / "features.pt")
    policy.set_gripper_bc_mode()
    policy.save_gripper_bc(tmp_path / "head.pt")
    policy.save_shared_features(tmp_path / "different_features.pt")
    with pytest.raises(ValueError, match="matching shared_feature_checkpoint"):
        policy.load_gripper_bc(tmp_path / "head.pt")
    torch.save({"format_version": 1, "state_dict": {}}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="independent"):
        policy.load_gripper_bc(tmp_path / "old.pt")


def test_factory_default_and_explicit_legacy(policy, monkeypatch):
    from dataclasses import asdict

    from omegaconf import OmegaConf

    from rlinf.models.embodiment.residual_policy import get_model
    from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
        SplitGripperConfig,
        SplitGripperPolicy,
    )

    monkeypatch.setattr(SharedGripperConfig, "_update_info", lambda self: None)
    cfg = asdict(policy.residual_cfg)
    cfg.pop("gripper_architecture")
    assert isinstance(get_model(OmegaConf.create(cfg)), SharedGripperPolicy)
    cfg["gripper_architecture"] = "independent"
    monkeypatch.setattr(SplitGripperConfig, "_update_info", lambda self: None)
    assert isinstance(get_model(OmegaConf.create(cfg)), SplitGripperPolicy)
