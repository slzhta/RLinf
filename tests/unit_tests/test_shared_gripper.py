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


def test_full_gripper_bc_updates_only_frontend_and_head(policy, tmp_path):
    obs = observation()
    policy.set_gripper_bc_mode(train_shared_features=True)
    prefixes = ("encoders.", "state_proj.", "gripper_head.")
    assert all(
        p.requires_grad == n.startswith(prefixes) for n, p in policy.named_parameters()
    )
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3
    )
    target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    F.binary_cross_entropy_with_logits(policy.gripper_bc_logits(obs), target).backward()
    for module in [policy.encoders, policy.state_proj, policy.gripper_head]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    optimizer.step()
    for prefix in prefixes:
        assert any(
            not torch.equal(before[k], v)
            for k, v in policy.state_dict().items()
            if k.startswith(prefix)
        )
    assert all(
        torch.equal(before[k], v)
        for k, v in policy.state_dict().items()
        if not k.startswith(prefixes)
    )
    policy.eval()
    feature_path, head_path = tmp_path / "features.pt", tmp_path / "head.pt"
    policy.save_gripper_bc_pair(feature_path, head_path)
    assert all(
        p.requires_grad == n.startswith(prefixes) for n, p in policy.named_parameters()
    )
    with pytest.raises(ValueError, match="Save/load shared features"):
        policy.save_gripper_bc(tmp_path / "stale_head.pt")
    restored = SharedGripperPolicy(deepcopy(policy.residual_cfg)).eval()
    untouched = {
        k: v.clone()
        for k, v in restored.state_dict().items()
        if not k.startswith(prefixes)
    }
    restored.load_shared_features(feature_path)
    restored.load_gripper_bc(head_path)
    torch.testing.assert_close(
        policy.gripper_bc_logits(obs), restored.gripper_bc_logits(obs)
    )
    assert all(torch.equal(restored.state_dict()[k], v) for k, v in untouched.items())
    restored.set_rl_mode()
    assert all(p.requires_grad for p in restored.parameters())
    assert not any(e.freeze_backbone for e in restored.encoders)


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


def make_separate_policy(policy, tmp_path):
    policy.set_gripper_bc_mode()
    features, head = tmp_path / "features.pt", tmp_path / "head.pt"
    policy.save_gripper_bc_pair(features, head)
    cfg = deepcopy(policy.residual_cfg)
    cfg.gripper_share_features = False
    cfg.shared_feature_checkpoint = str(features)
    cfg.gripper_checkpoint = str(head)
    separate = SharedGripperPolicy(cfg)
    result = separate.load_state_dict(policy.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(
        k.startswith(("gripper_encoders.", "gripper_state_proj."))
        for k in result.missing_keys
    )
    separate.eval()
    policy.set_rl_mode()
    policy.eval()
    return separate


def test_separate_initial_outputs_and_checkpoint_roundtrip(policy, tmp_path):
    separate = make_separate_policy(policy, tmp_path)
    obs = observation()
    for original, cloned in zip(
        distribution_outputs(policy, obs), distribution_outputs(separate, obs)
    ):
        torch.testing.assert_close(original, cloned, rtol=0, atol=0)
    for a, b in [
        (policy.encoders, separate.gripper_encoders),
        (separate.encoders, separate.gripper_encoders),
        (separate.state_proj, separate.gripper_state_proj),
    ]:
        assert {p.data_ptr() for p in a.parameters()}.isdisjoint(
            {p.data_ptr() for p in b.parameters()}
        )
    actions, rollout = separate.predict_action_batch(obs, mode="train")
    replay = separate.default_forward(rollout["forward_inputs"])
    torch.testing.assert_close(
        replay["logprobs"], rollout["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    assert (actions[..., 6].abs() == 1).all()
    with torch.no_grad():
        separate.gripper_state_proj[0].weight.add_(0.25)
    full = tmp_path / "full.pt"
    torch.save(separate.state_dict(), full)
    restored = SharedGripperPolicy(deepcopy(separate.residual_cfg)).eval()
    restored.load_state_dict(torch.load(full, weights_only=True), strict=True)
    torch.testing.assert_close(
        restored.gripper_bc_logits(obs), separate.gripper_bc_logits(obs)
    )
    assert not torch.equal(
        restored.state_proj[0].weight, restored.gripper_state_proj[0].weight
    )
    with pytest.raises(RuntimeError):
        separate.load_state_dict(policy.state_dict(), strict=True)


def test_value_gradient_isolated_but_ppo_updates_separate_gripper(policy, tmp_path):
    separate = make_separate_policy(policy, tmp_path)
    _, rollout = separate.predict_action_batch(observation(), mode="train")
    result = separate.default_forward(rollout["forward_inputs"])
    result["values"].square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in separate.encoders.parameters()
    )
    assert all(
        p.grad is None
        for n, p in separate.named_parameters()
        if n.startswith("gripper_")
    )
    separate.zero_grad(set_to_none=True)
    before = {k: v.clone() for k, v in separate.state_dict().items()}
    replay = separate.default_forward(rollout["forward_inputs"])
    advantage = torch.tensor([1.0, -1.0, 0.5, -0.5])
    ratio = (replay["logprobs"][..., 6] - rollout["prev_logprobs"][..., 6]).exp()
    loss = -(ratio * advantage).mean()
    loss.backward()
    for module in [
        separate.gripper_encoders,
        separate.gripper_state_proj,
        separate.gripper_head,
    ]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for n, p in separate.named_parameters()
        if not n.startswith("gripper_")
    )
    torch.optim.AdamW(separate.parameters(), lr=1e-3, weight_decay=0).step()
    assert any(
        not torch.equal(v, separate.state_dict()[k])
        for k, v in before.items()
        if k.startswith("gripper_encoders.")
    )
    assert all(
        torch.equal(v, separate.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("gripper_")
    )


def test_separate_bc_trains_and_exports_only_gripper_copy(policy, tmp_path):
    separate = make_separate_policy(policy, tmp_path)
    separate.set_gripper_bc_mode(train_shared_features=True)
    assert all(
        p.requires_grad == n.startswith("gripper_")
        for n, p in separate.named_parameters()
    )
    before = {k: v.clone() for k, v in separate.state_dict().items()}
    obs = observation()
    F.binary_cross_entropy_with_logits(
        separate.gripper_bc_logits(obs), torch.ones(4, 1)
    ).backward()
    torch.optim.AdamW(
        [p for p in separate.parameters() if p.requires_grad], lr=1e-3
    ).step()
    assert all(
        torch.equal(v, separate.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("gripper_")
    )
    separate.save_gripper_bc_pair(
        tmp_path / "separate_features.pt", tmp_path / "separate_head.pt"
    )
    separate.eval()
    restored = SharedGripperPolicy(deepcopy(separate.residual_cfg)).eval()
    restored.load_shared_features(tmp_path / "separate_features.pt")
    restored.load_gripper_bc(tmp_path / "separate_head.pt")
    torch.testing.assert_close(
        restored.gripper_bc_logits(obs), separate.gripper_bc_logits(obs)
    )
    separate.set_rl_mode()
    assert all(p.requires_grad for p in separate.parameters())
    assert not any(e.freeze_backbone for e in separate.gripper_encoders)
