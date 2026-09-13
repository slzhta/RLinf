# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Six-axis residual replay, gradient and task-input contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.residual_policy import get_model
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg
from rlinf.workers.rollout.hf.residual_inference import RolloutResidualInference


@pytest.fixture
def cfg():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        return compose(
            config_name="sim_rl_peg_insertion_openpi_residual_wrist_state_n32_ue4"
        )


def test_peg_config(cfg):
    validate_residual_cfg(cfg)
    assert cfg.actor.model.action_dim == cfg.base_model.action_dim == 6
    assert cfg.peg_task.success_xy == 0.001
    assert cfg.runner.val_check_interval < 0


def test_peg_co_training_config():
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(
            config_name="co_rl_peg_insertion_openpi_residual_wrist_state_n32_ue4"
        )
    validate_residual_cfg(cfg)
    assert cfg.algorithm.co_training_rollout_routing_mode == "paired"
    assert cfg.algorithm.co_training_domain_buffer.mode == "latest"
    real = cfg.env.train.co_training_env_cfg
    assert list(real.state_keys) == ["ee_target_delta"]
    assert real.override_cfg.peg_config.target_ee_pose == cfg.peg_task.target_ee_pose
    assert real.override_cfg.peg_config.success_xy == 0.01
    assert cfg.peg_task.success_xy == 0.001
    samples = (
        cfg.algorithm.co_training_domain_buffer.sample_rollout_length
        * cfg.env.train.max_steps_per_rollout_epoch
    )
    assert samples % cfg.actor.global_batch_size == 0
    assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0


def test_peg_encoder_path_resolves_on_worker(cfg, monkeypatch, tmp_path):
    from rlinf.models.embodiment.modules.resnet_utils import ResNet10

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    torch.save(ResNet10().state_dict(), tmp_path / "resnet10_pretrained.pt")
    cfg.actor.model.model_path = "~"
    policy = get_model(cfg.actor.model)
    assert policy is not None
    assert cfg.actor.model.model_path == "~"


@pytest.mark.parametrize(
    "key,value",
    [
        ("actor.model.state_dim", 14),
        ("actor.model.image_num", 2),
        ("actor.model.gripper_checkpoint", "/unused.pt"),
        ("base_model.openpi.discrete_state_input", False),
        ("peg_task.dense_reward_scale", 0.1),
        ("env.train.init_params.id", "PickAndPlaceDigitalTwin-v1"),
        ("runner.val_check_interval", 25),
    ],
)
def test_peg_rejects_mismatches(cfg, key, value):
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError):
        validate_residual_cfg(cfg)


def test_peg_replay_updates_features_and_preserves_frozen_backbone(cfg, tmp_path):
    from rlinf.models.embodiment.modules.resnet_utils import ResNet10

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.save(ResNet10().state_dict(), tmp_path / "resnet10_pretrained.pt")
    cfg.actor.model.model_path = str(tmp_path)
    policy = get_model(cfg.actor.model)
    assert not hasattr(policy, "gripper_head")
    assert policy.state_proj[0].in_features == 6
    assert policy.action_proj[0].in_features == 8 * 7
    observation = {
        "main_images": torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        "states": torch.randn(2, 6),
        "_residual_reset_mask": torch.zeros(2, dtype=torch.bool),
    }

    class Base:
        def predict_action_batch(self, env_obs, **kwargs):
            return torch.full((len(env_obs["states"]), 8, 6), 0.1), {}

    inference = RolloutResidualInference(cfg, "cpu", base_policy=Base())
    deterministic, _ = inference.predict(policy, observation, mode="eval")
    torch.testing.assert_close(deterministic, torch.full((2, 1, 6), 0.1))
    executed, info = inference.predict(policy, observation, mode="train")
    inputs = info["forward_inputs"]
    assert inputs["states"].shape == (2, 62)
    torch.testing.assert_close(inputs["states"][:, -6:], observation["states"])
    sample = inputs["action"].clone()
    torch.testing.assert_close(executed[:, 0], (0.1 + 0.2 * sample).clamp(-1, 1))
    replay = policy.default_forward(inputs)
    torch.testing.assert_close(
        replay["logprobs"], info["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    assert replay["entropy"].shape == (2, 1, 6)
    ratio = (replay["logprobs"] - info["prev_logprobs"]).sum(-1).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    trainable = [
        policy.actor_mean.weight,
        policy.state_proj[0].weight,
        policy.action_proj[0].weight,
        next(policy.value_head.parameters()),
    ]
    before = [p.detach().clone() for p in trainable]
    loss = -replay["logprobs"].mean() + replay["values"].square().mean()
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=1e-3
    )
    loss.backward()
    for parameter in trainable:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(
        not p.requires_grad and p.grad is None
        for p in policy.encoders[0].resnet_backbone.parameters()
    )
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, trainable))
    torch.testing.assert_close(inputs["action"], sample)


def test_peg_openpi_transforms():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.peg_policy import PegInputs, PegOutputs

    image = np.full((224, 224, 3), 127, dtype=np.uint8)
    state = np.arange(6, dtype=np.float32) / 10
    data = PegInputs()(
        {
            "observation/image": image,
            "observation/state": state,
            "actions": np.ones((8, 6)),
            "prompt": "Insert the peg",
        }
    )
    np.testing.assert_array_equal(data["state"], state)
    np.testing.assert_array_equal(data["image"]["base_0_rgb"], image)
    assert list(data["image_mask"].values()) == [True, False, False]
    assert data["actions"].shape == (8, 32)
    assert data["prompt"] == "Insert the peg"
    np.testing.assert_array_equal(PegOutputs()(data)["actions"], np.ones((8, 6)))
