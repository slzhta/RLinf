# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for rollout-side residual cache and PPO execution separation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg
from rlinf.workers.rollout.hf.residual_inference import RolloutResidualInference


class Base:
    def __init__(self):
        self.calls = []

    def predict_action_batch(self, env_obs, **kwargs):
        self.calls.append(env_obs["states"][:, 0].clone())
        b = len(env_obs["main_images"])
        plan = torch.zeros(b, 3, 7)
        plan[:, :, :6] = torch.tensor([0.1, 0.2, 0.3])[None, :, None]
        plan[:, :, 6] = 0.9
        return plan, {}


class Policy:
    def predict_action_batch(self, env_obs, **kwargs):
        b = len(env_obs["main_images"])
        actions = torch.full((b, 1, 7), 2.0)
        actions[:, :, 6] = -1
        return actions, {
            "forward_inputs": {
                "action": actions[:, 0].clone(),
                "condition": env_obs["base_actions"].clone(),
            },
            "prev_logprobs": torch.full((b, 7), -0.7),
        }


def obs(reset=(False, False)):
    return {
        "main_images": torch.zeros(2, 8, 8, 3),
        "states": torch.tensor([[1.0], [2.0]]),
        "_residual_reset_mask": torch.tensor(reset),
    }


def adapter():
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "base_horizon": 3,
                    "action_dim": 7,
                    "residual_action_scale": [0.4] * 6 + [0.0],
                }
            }
        }
    )
    base = Base()
    return RolloutResidualInference(cfg, "cpu", base_policy=base), base


def test_composition_preserves_sample_and_logprob():
    inference, base = adapter()
    executed, result = inference.predict(Policy(), obs())
    torch.testing.assert_close(executed[:, :, :6], torch.full((2, 1, 6), 0.5))
    assert (executed[:, :, 6] == -1).all()
    assert (result["forward_inputs"]["action"][:, :6] == 2).all()
    assert (result["prev_logprobs"] == -0.7).all()
    saved = result["forward_inputs"]["condition"].clone()
    for _ in range(4):
        inference.predict(Policy(), obs())
    torch.testing.assert_close(saved, result["forward_inputs"]["condition"])
    assert len(base.calls) == 2


def test_partial_reset_only_replans_ended_environment():
    inference, base = adapter()
    inference.predict(Policy(), obs())
    executed, _ = inference.predict(Policy(), obs((True, False)))
    torch.testing.assert_close(base.calls[-1], torch.tensor([1.0]))
    torch.testing.assert_close(executed[:, 0, 0], torch.tensor([0.5, 0.6]))


def test_epoch_value_request_does_not_skip_base_action():
    inference, base = adapter()
    inference.predict(Policy(), obs())
    inference.predict(Policy(), obs(), consume=False)
    executed, _ = inference.predict(Policy(), obs())
    torch.testing.assert_close(executed[:, 0, 0], torch.tensor([0.6, 0.6]))
    assert len(base.calls) == 1


def test_domain_workers_have_independent_caches():
    real, _ = adapter()
    sim, _ = adapter()
    real.predict(Policy(), obs())
    real.predict(Policy(), obs())
    sim.predict(Policy(), obs())
    assert real.caches["train"].position.tolist() == [2, 2]
    assert sim.caches["train"].position.tolist() == [1, 1]


def test_missing_reset_signal_is_rejected():
    inference, _ = adapter()
    observation = obs()
    del observation["_residual_reset_mask"]
    with pytest.raises(ValueError, match="reset mask"):
        inference.predict(Policy(), observation)


def test_real_joint_policy_replays_after_composition(tmp_path):
    from rlinf.models.embodiment.modules.resnet_utils import ResNet10
    from rlinf.models.embodiment.residual_policy.split_gripper_policy import (
        SplitGripperConfig,
        SplitGripperPolicy,
    )

    torch.set_num_threads(2)
    torch.save(ResNet10().state_dict(), tmp_path / "encoder.pt")
    model_cfg = SplitGripperConfig()
    model_cfg.update_from_dict(
        {
            "model_path": str(tmp_path),
            "encoder_config": {"ckpt_name": "encoder.pt", "dropout": 0.0},
            "use_state": True,
            "state_dim": 14,
            "image_num": 2,
            "image_size": [3, 32, 32],
            "action_dim": 7,
            "base_horizon": 3,
            "add_value_head": True,
            "add_q_head": False,
            "gripper": {
                "image_num": 2,
                "image_size": 32,
                "state_dim": 14,
                "hidden_dim": 32,
            },
        }
    )
    policy = SplitGripperPolicy(model_cfg)
    observation = {
        "main_images": torch.zeros(2, 32, 32, 3, dtype=torch.uint8),
        "extra_view_images": torch.zeros(2, 1, 32, 32, 3, dtype=torch.uint8),
        "states": torch.randn(2, 14),
        "_residual_reset_mask": torch.zeros(2, dtype=torch.bool),
    }
    inference, _ = adapter()
    executed, result = inference.predict(policy, observation, return_obs=True)
    saved = result["forward_inputs"]
    torch.testing.assert_close(saved["states"][:, -14:], observation["states"])
    torch.testing.assert_close(
        policy._gripper_obs(saved)["states"], observation["states"]
    )
    replay = policy.default_forward(saved)
    torch.testing.assert_close(
        replay["logprobs"], result["prev_logprobs"], atol=1e-5, rtol=1e-5
    )
    assert (saved["action"][:, :6].abs() <= 1).all()
    assert replay["entropy"].shape == (2, 1, 7)
    assert not torch.equal(executed[:, 0, :6], saved["action"][:, :6])
    loss = -replay["logprobs"].sum()
    loss.backward()
    assert policy.actor_mean.weight.grad.abs().sum() > 0
    assert policy.gripper.head[-1].weight.grad.abs().sum() > 0


def test_base_checkpoint_resolves_on_rollout_host(monkeypatch, tmp_path):
    from rlinf.models.embodiment import openpi

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    loaded = []
    base = torch.nn.Linear(1, 1)

    def get_model(cfg):
        loaded.append(cfg.model_path)
        return base

    monkeypatch.setattr(openpi, "get_model", get_model)
    cfg = OmegaConf.create(
        {
            "base_model": {"model_path": "~/wangyinghan/models/base"},
            "actor": {"model": {"residual_action_scale": [0.4] * 6 + [0.0]}},
        }
    )
    RolloutResidualInference(cfg, "cpu")
    assert loaded == [str(tmp_path / "wangyinghan/models/base")]
    assert cfg.base_model.model_path == "~/wangyinghan/models/base"
    assert not base.training
    assert all(not p.requires_grad for p in base.parameters())


def composed_config(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        cfg = compose(
            config_name="co_rl_pick_and_place_async_ppo_residual_cnn_gripper_paired"
        )
    cfg.checkpoints = {
        "openpi": "/test/base",
        "encoder": "/test/encoder",
        "gripper": "/test/gripper.pt",
    }
    return cfg


def test_paired_configuration_validates(monkeypatch):
    cfg = composed_config(monkeypatch)
    validate_residual_cfg(cfg)
    assert cfg.cluster.component_placement.env.placement == "0:0,2:1"
    assert cfg.cluster.component_placement.rollout.placement == "0-1"
    assert cfg.algorithm.co_training_domain_buffer.mode == "latest"
    for domain in (cfg.env.train, cfg.env.eval):
        assert list(domain.co_training_env_cfg.state_keys) == [
            "arm_joint_position",
            "tcp_pose",
            "gripper_open_state",
        ]


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("algorithm.co_training_rollout_routing_mode", "single", "paired"),
        ("algorithm.bootstrap_type", "always", "bootstrap_type"),
        ("env.train.co_training_env_cfg.use_spacemouse", True, "intervention"),
        ("runner.only_eval", True, "without evaluation"),
        ("actor.sync_weight_no_wait", True, "synchronization"),
        ("rollout.residual_base_inference", False, "rollout-side"),
        ("rollout.expert_model.model_path", "/test/expert", "expert"),
        ("env.train.residual.enabled", True, "env-side"),
        (
            "env.train.co_training_env_cfg.state_keys",
            ["tcp_pose", "arm_joint_position", "gripper_open_state"],
            "state_keys",
        ),
    ],
)
def test_unsupported_modes_fail_before_workers(monkeypatch, key, value, match):
    cfg = composed_config(monkeypatch)
    OmegaConf.update(cfg, key, value, force_add=True)
    with pytest.raises(ValueError, match=match):
        validate_residual_cfg(cfg)


def load_method(relative_path, class_name, method_name):
    """Exercise worker adapters without allocating Ray workers or OpenPI weights."""
    path = Path(__file__).resolve().parents[2] / relative_path
    module = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method_name]


def test_reset_signal_does_not_modify_original_observations():
    method = load_method(
        "rlinf/workers/env/env_worker.py", "EnvWorker", "_residual_rollout_obs"
    )
    worker = SimpleNamespace(cfg=OmegaConf.create({"rollout": {}}))
    batch = {
        "obs": {"states": torch.zeros(2, 14)},
        "dones": torch.tensor([[False], [True]]),
    }
    assert method(worker, batch) is batch["obs"]
    worker.cfg.rollout.residual_base_inference = True
    result = method(worker, batch)
    assert result["_residual_reset_mask"].tolist() == [False, True]
    assert "_residual_reset_mask" not in batch["obs"]


def test_pnp_wrist_mapping_preserves_other_openpi_configs():
    path = "rlinf/models/embodiment/openpi/openpi_action_model.py"
    tree = ast.parse(
        (Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8")
    )
    name = next(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(m, ast.FunctionDef) and m.name == "obs_processor"
            for m in node.body
        )
    )
    method = load_method(path, name, "obs_processor")
    model = SimpleNamespace(config=SimpleNamespace(config_name="pi05_pnp"))
    observation = {
        "main_images": torch.zeros(2, 8, 8, 3),
        "extra_view_images": torch.ones(2, 1, 8, 8, 3),
        "wrist_images": None,
        "states": torch.randn(2, 14),
        "task_descriptions": ["pnp", "pnp"],
    }
    mapped = method(model, observation)
    torch.testing.assert_close(
        mapped["observation/wrist_image"], observation["extra_view_images"][:, 0]
    )
    assert mapped["observation/state"] is observation["states"]
    assert mapped["prompt"] is observation["task_descriptions"]
    model.config.config_name = "pi05_franka"
    assert "observation/wrist_image" not in method(model, observation)
    observation["wrist_images"] = torch.zeros(2, 8, 8, 3)
    assert (
        method(model, observation)["observation/wrist_image"]
        is observation["wrist_images"]
    )
