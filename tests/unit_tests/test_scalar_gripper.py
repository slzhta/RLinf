# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Scalar gripper BC and action-contract regression tests, without hardware."""

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.residual_policy.gripper_cnn import (
    GripperCNN,
    GripperCNNConfig,
)


@pytest.fixture
def scalar_policy(tmp_path):
    from rlinf.models.embodiment.modules.resnet_utils import ResNet10
    from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
        SplitGripperConfig,
        SplitGripperPolicy,
    )

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.save(ResNet10().state_dict(), tmp_path / "encoder.pt")
    cfg = SplitGripperConfig()
    cfg.update_from_dict(
        {
            "model_path": str(tmp_path),
            "encoder_config": {"ckpt_name": "encoder.pt", "dropout": 0.0},
            "image_size": [3, 32, 32],
            "image_num": 2,
            "state_dim": 14,
            "use_state": True,
            "action_dim": 7,
            "base_horizon": 3,
            "add_value_head": True,
            "gripper": {
                "image_num": 2,
                "image_size": 32,
                "state_dim": 14,
                "hidden_dim": 32,
                "output_mode": "scalar",
            },
        }
    )
    return SplitGripperPolicy(cfg)


def policy_obs():
    return {
        "main_images": torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (4, 1, 32, 32, 3), dtype=torch.uint8
        ),
        "states": torch.randn(4, 14),
        "base_actions": torch.zeros(4, 3, 7),
        "base_action_mask": torch.ones(4, 3, dtype=torch.bool),
    }


def test_scalar_joint_ppo_replay_and_updates(scalar_policy):
    model = scalar_policy
    actions, result = model.predict_action_batch(policy_obs(), mode="train")
    inputs = result["forward_inputs"]
    torch.testing.assert_close(actions[:, 0, 6], inputs["action"][:, 6])
    assert (actions[:, :, 6].abs() < 1).all()
    replay = model.default_forward(inputs)
    torch.testing.assert_close(
        replay["logprobs"], result["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    assert replay["entropy"].shape == (4, 1, 7)
    ratio = (replay["logprobs"].sum(-1) - result["prev_logprobs"].sum(-1)).exp()
    advantage = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tracked = [
        model.gripper.head[-1].weight,
        model.gripper.encoder[0].weight,
        model.gripper_logstd,
        model.actor_mean.weight,
    ]
    before = [p.detach().clone() for p in tracked]
    loss.backward()
    for parameter in tracked:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    for parameter, old in zip(tracked, before):
        assert not torch.equal(parameter, old)


def test_scalar_eval_mean_and_train_eval_independent_caches(scalar_policy):
    from omegaconf import OmegaConf

    from rlinf.workers.rollout.hf.residual_inference import RolloutResidualInference

    class Base:
        def predict_action_batch(self, env_obs, **kwargs):
            return torch.ones(len(env_obs["states"]), 3, 7), {}

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "base_horizon": 3,
                    "action_dim": 7,
                    "residual_action_scale": [0.4] * 6 + [0.0],
                    "gripper": {"output_mode": "scalar"},
                }
            }
        }
    )
    inference = RolloutResidualInference(cfg, "cpu", base_policy=Base())
    with torch.no_grad():
        scalar_policy.gripper.head[-1].weight.zero_()
        scalar_policy.gripper.head[-1].bias.fill_(torch.atanh(torch.tensor(0.2)))
    obs = policy_obs()
    obs["_residual_reset_mask"] = torch.zeros(4, dtype=torch.bool)
    executed, result = inference.predict(scalar_policy, obs, mode="eval")
    torch.testing.assert_close(executed[:, 0, 6], torch.full((4,), 0.2))
    torch.testing.assert_close(
        result["forward_inputs"]["action"][:, 6], executed[:, 0, 6]
    )
    inference.predict(scalar_policy, obs, mode="train")
    assert inference.caches["eval"].position.tolist() == [1] * 4
    assert inference.caches["train"].position.tolist() == [1] * 4
    second, _ = inference.predict(scalar_policy, obs, mode="eval")
    torch.testing.assert_close(second[:, 0, 6], executed[:, 0, 6])


def test_scalar_logstd_bounds_and_checkpoint_load(scalar_policy, tmp_path):
    from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
        SplitGripperPolicy,
    )

    model = scalar_policy
    model.gripper.save_bc(tmp_path / "scalar.pt")
    cfg = model.residual_cfg
    cfg.gripper_checkpoint = str(tmp_path / "scalar.pt")
    loaded = SplitGripperPolicy(cfg)
    torch.testing.assert_close(loaded.gripper_logstd, torch.tensor([[-1.0]]))
    for key, value in model.gripper.state_dict().items():
        torch.testing.assert_close(loaded.gripper.state_dict()[key], value)
    with torch.no_grad():
        loaded.gripper_logstd.fill_(5)
    distribution = loaded._gripper_distribution(torch.zeros(2, 1))
    torch.testing.assert_close(distribution.stddev, torch.ones(2, 1))
    with torch.no_grad():
        loaded.gripper_logstd.fill_(-10)
    torch.testing.assert_close(
        loaded._gripper_distribution(torch.zeros(2, 1)).stddev,
        torch.full((2, 1), np.exp(-4), dtype=torch.float32),
    )
    cfg.gripper_initial_logstd = 1.0
    with pytest.raises(ValueError, match="gripper_initial_logstd"):
        SplitGripperPolicy(cfg)


@pytest.mark.parametrize("mode", ["bernoulli", "scalar"])
def test_bc_checkpoint_roundtrip_and_mode_mismatch(tmp_path, mode):
    cfg = GripperCNNConfig(output_mode=mode)
    model = GripperCNN(cfg)
    path = tmp_path / "bc.pt"
    model.save_bc(path)
    restored = GripperCNN(cfg)
    restored.load_bc(path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    other = "scalar" if mode == "bernoulli" else "bernoulli"
    with pytest.raises(ValueError, match="configuration/convention mismatch"):
        GripperCNN(GripperCNNConfig(output_mode=other)).load_bc(path)


def test_legacy_bc_remains_bernoulli(tmp_path):
    model = GripperCNN(GripperCNNConfig())
    path = tmp_path / "legacy.pt"
    model.save_bc(path)
    saved = torch.load(path, weights_only=True)
    del saved["config"]["output_mode"]
    torch.save(saved, path)
    model.load_bc(path)
    with pytest.raises(ValueError, match="configuration/convention mismatch"):
        GripperCNN(GripperCNNConfig(output_mode="scalar")).load_bc(path)
