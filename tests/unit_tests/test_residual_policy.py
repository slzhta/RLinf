# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.residual import BaseActionCache
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy
from rlinf.models.embodiment.modules.resnet_utils import ResNet10
from rlinf.models.embodiment.residual_policy.residual_policy import (
    ResidualConfig,
    ResidualPolicy,
)
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/embodiment/config"


@pytest.mark.parametrize("boundary,direction", [(0.0, -1.0), (-5.0, 1.0)])
def test_slz_logstd_forward_bounds_and_known_saturation(encoder_path, boundary, direction):
    """Document SLZ's forward-only clamp, including its out-of-range zero gradient.

    TODO(agent): SLZ does not project raw optimizer parameters. This test records
    the selected baseline; it does not claim that saturation has been repaired.
    """
    cfg = model_config(encoder_path)
    cfg.logstd_range = [-5.0, 0.0]
    policy = ResidualPolicy(cfg)
    with torch.no_grad():
        policy.actor_logstd.fill_(boundary)
    optimizer = torch.optim.SGD([policy.actor_logstd], lr=0.1)
    (direction * policy.actor_logstd.sum()).backward()
    optimizer.step()
    torch.testing.assert_close(policy.actor_logstd, torch.full_like(policy.actor_logstd, boundary - 0.1 * direction))
    optimizer.zero_grad()
    std = policy._action_std(policy.actor_logstd)
    torch.testing.assert_close(std.log(), torch.full_like(std, boundary))
    std.sum().backward()
    assert torch.count_nonzero(policy.actor_logstd.grad) == 0


def test_slz_checkpoint_preserves_raw_logstd_and_bounds_forward(encoder_path):
    cfg = model_config(encoder_path)
    cfg.logstd_range = [-5.0, 0.0]
    policy = ResidualPolicy(cfg)
    weights = policy.state_dict()
    raw = torch.tensor([[0.3, -5.4, -0.2, 0.1, -2.0, 0.0]])
    weights["actor_logstd"] = raw
    policy.load_state_dict(weights)
    torch.testing.assert_close(policy.actor_logstd, raw)
    torch.testing.assert_close(policy._action_std(policy.actor_logstd).log(), raw.clamp(-5.0, 0.0))


@pytest.fixture(scope="module")
def encoder_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("residual_encoder")
    torch.save(ResNet10().state_dict(), path / "resnet10_pretrained.pt")
    return path


def model_config(encoder_path, use_state=False):
    cfg = ResidualConfig()
    cfg.update_from_dict(
        {
            "model_path": str(encoder_path),
            "encoder_config": {"ckpt_name": "resnet10_pretrained.pt", "dropout": 0.0},
            "use_state": use_state,
            "state_dim": 14,
            "image_size": [3, 32, 32],
            "image_num": 2,
            "action_dim": 7,
            "base_horizon": 3,
            "add_value_head": True,
            "add_q_head": False,
            "initial_logstd": -3.0,
            "logstd_range": [-5.0, -1.0],
        }
    )
    return cfg


def observation(batch=2, use_state=False):
    obs = {
        "main_images": torch.randint(0, 256, (batch, 32, 32, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (batch, 1, 32, 32, 3), dtype=torch.uint8
        ),
        "base_actions": torch.randn(batch, 3, 7),
        "base_action_mask": torch.ones(batch, 3, dtype=torch.bool),
    }
    if use_state:
        obs["states"] = torch.randn(batch, 14)
    return obs


def test_state_disabled_ignores_states_and_initial_mean_is_zero(encoder_path):
    cfg = model_config(encoder_path)
    policy = ResidualPolicy(cfg)
    assert cfg.use_state is False and cfg.action_dim == 7 and cfg.state_dim == 14
    assert policy.cfg.use_state is True and policy.cfg.state_dim == 24
    obs = observation()
    actions, _ = policy.predict_action_batch(obs, mode="eval")
    assert torch.equal(actions, torch.zeros_like(actions))
    features = policy._prepare_cnn_obs(obs)["states"]
    obs["states"] = torch.full((2, 14), float("nan"))
    torch.testing.assert_close(policy._prepare_cnn_obs(obs)["states"], features)
    chunk_actions, result = policy.predict_action_batch(obs)
    actions = chunk_actions[:, 0]
    logprobs = result["prev_logprobs"]
    assert actions.shape == (2, 7)
    assert torch.equal(actions[:, 6], torch.zeros(2))
    assert torch.equal(logprobs[:, 6], torch.zeros(2))
    assert torch.isfinite(logprobs).all()


def test_state_enabled_changes_features_and_requires_correct_shape(encoder_path):
    policy = ResidualPolicy(model_config(encoder_path, True))
    obs = observation(use_state=True)
    first = policy._prepare_cnn_obs(obs)["states"]
    obs["states"] = obs["states"] + 5
    assert not torch.allclose(policy._prepare_cnn_obs(obs)["states"], first)
    obs.pop("states")
    with pytest.raises(ValueError, match="use_state"):
        policy.predict_action_batch(obs)


def test_missing_camera_and_base_conditions_fail_explicitly(encoder_path):
    policy = ResidualPolicy(model_config(encoder_path))
    obs = observation()
    obs.pop("extra_view_images")
    with pytest.raises(ValueError, match="camera"):
        policy.predict_action_batch(obs)
    obs = observation()
    obs["base_actions"] = obs["base_actions"][:, :1]
    with pytest.raises(ValueError, match="base_actions"):
        policy.predict_action_batch(obs)


class FakeBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.batch_sizes = []

    def predict_action_batch(self, env_obs, mode, compute_values):
        assert not torch.is_grad_enabled()
        assert mode == "eval" and not compute_values
        # Base state must remain available even for a state-free residual.
        batch = len(env_obs["states"])
        self.batch_sizes.append(batch)
        actions = torch.arange(21).reshape(1, 3, 7).float() / 30 + self.weight
        return actions.repeat(batch, 1, 1), {}


def test_cache_zero_residual_chunk_boundary_and_partial_reset():
    base = FakeBase()
    cache = BaseActionCache(2, 3, 7, torch.device("cpu"))
    obs = {"states": torch.zeros(2, 14), "task_descriptions": ["a", "b"]}
    cache.refresh(obs, base)
    saved = cache.observation()
    original = cache.actions.clone()
    zeros = torch.zeros(2, 7)
    scale = torch.tensor([0.1] * 6 + [0.0])
    torch.testing.assert_close(cache.compose(zeros, scale), original[:, 0])
    cache.refresh(obs, base)
    assert base.batch_sizes == [2]
    torch.testing.assert_close(saved["base_actions"], original)
    assert cache.observation()["base_action_mask"].tolist() == [[True, True, False]] * 2
    cache.invalidate(torch.tensor([0]))
    cache.refresh(obs, base)
    assert base.batch_sizes == [2, 1]
    assert cache.position.tolist() == [0, 1]
    expected = torch.stack([original[0, 0], original[1, 1]])
    torch.testing.assert_close(cache.compose(zeros, scale), expected)
    cache.compose(zeros, scale)
    cache.refresh(obs, base)
    assert base.batch_sizes == [2, 1, 1]


def test_cache_limits_composition_and_preserves_gripper():
    cache = BaseActionCache(1, 3, 7, torch.device("cpu"))
    cache.refresh({"states": torch.zeros(1, 14)}, FakeBase())
    cache.actions[0, 0, 0] = 0.99
    base_action = cache.actions[:, 0].clone()
    actual = cache.compose(torch.full((1, 7), 5.0), torch.tensor([0.1] * 6 + [0.0]))
    assert actual[0, 0] == 1
    assert actual[0, 6] == base_action[0, 6]
    torch.testing.assert_close(actual[:, 1:6], base_action[:, 1:6] + 0.1)
    with pytest.raises(ValueError, match="finite"):
        cache.compose(torch.full((1, 7), float("nan")), torch.ones(7))


def test_env_output_retains_residual_plan_without_changing_legacy_keys():
    obs = observation()
    prepared = EnvOutput(obs=obs).to_dict()["obs"]
    torch.testing.assert_close(prepared["base_actions"], obs["base_actions"])
    torch.testing.assert_close(prepared["base_action_mask"], obs["base_action_mask"])
    assert prepared["states"] is None
    legacy = EnvOutput(obs={"main_images": obs["main_images"]}).to_dict()["obs"]
    assert "base_actions" not in legacy


@pytest.mark.parametrize("use_state", [False, True])
def test_ppo_recomputes_rollout_likelihood_and_updates_actor_and_value(
    encoder_path, use_state
):
    from rlinf.algorithms.losses import compute_decoupled_ppo_actor_critic_loss

    policy = ResidualPolicy(model_config(encoder_path, use_state=use_state))
    obs = observation(batch=4, use_state=use_state)
    actions, rollout = policy.predict_action_batch(obs)
    policy.train()
    output = policy(forward_inputs=rollout["forward_inputs"])
    torch.testing.assert_close(output["logprobs"], rollout["prev_logprobs"])
    torch.testing.assert_close(output["values"], rollout["prev_values"])
    assert torch.equal(actions[:, 0, 6], torch.zeros(4))
    assert output["entropy"].shape == (4, 1, 7)
    assert torch.equal(output["entropy"][:, 0, 6], torch.zeros(4))
    assert not hasattr(policy, "q_head")
    assert not any("openpi" in key for key in policy.state_dict())
    before_actor = policy.actor_mean.weight.detach().clone()
    before_value = next(policy.value_head.parameters()).detach().clone()
    advantages = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])
    loss, metrics = compute_decoupled_ppo_actor_critic_loss(
        logprobs=output["logprobs"].sum(-1, keepdim=True),
        old_logprobs=rollout["prev_logprobs"].sum(-1, keepdim=True),
        advantages=advantages,
        values=output["values"],
        prev_values=rollout["prev_values"],
        returns=rollout["prev_values"] + advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        value_loss_coef=0.25,
        huber_delta=10.0,
    )
    loss = loss - 0.001 * output["entropy"].sum(-1).mean()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before_actor, policy.actor_mean.weight)
    assert not torch.equal(before_value, next(policy.value_head.parameters()))
    assert "actor/policy_loss" in metrics


def test_unbounded_gaussian_likelihood_uses_native_saved_action(encoder_path):
    policy = ResidualPolicy(model_config(encoder_path))
    with torch.no_grad():
        policy.actor_mean.bias.fill_(15)
    actions, rollout = policy.predict_action_batch(observation(), mode="train")
    output = policy(forward_inputs=rollout["forward_inputs"])
    assert (actions[..., :6] > 1).all()
    assert "raw_action" not in rollout["forward_inputs"]
    assert torch.isfinite(output["logprobs"]).all()
    torch.testing.assert_close(output["logprobs"], rollout["prev_logprobs"])


def test_ppo_rollout_inputs_are_independent_and_state_free(encoder_path):
    policy = ResidualPolicy(model_config(encoder_path))
    obs = observation(use_state=True)
    _, rollout = policy.predict_action_batch(obs)
    assert rollout["forward_inputs"]["states"].shape == (2, 24)
    saved = rollout["forward_inputs"]["states"].clone()
    obs["base_actions"].zero_()
    torch.testing.assert_close(rollout["forward_inputs"]["states"], saved)
    with pytest.raises(NotImplementedError, match="PPO"):
        policy(forward_type=ForwardType.SAC, obs=obs)


def test_hydra_config_and_contract_validation(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG.parent))
    with initialize_config_dir(config_dir=str(CONFIG), version_base="1.1"):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_a4")
    OmegaConf.resolve(cfg)
    validate_residual_cfg(cfg)
    assert not cfg.base_model.openpi.discrete_state_input
    assert cfg.base_model.openpi.config_name == "pi05_pnp"
    assert cfg.base_model.openpi.action_chunk == 10
    assert cfg.base_model.openpi.num_steps == 10
    assert cfg.base_model.openpi.action_env_dim == 7
    assert cfg.base_model.openpi.add_value_head is False
    assert cfg.actor.model.num_action_chunks == 1
    assert cfg.algorithm.loss_type == "decoupled_actor_critic"
    assert cfg.algorithm.adv_type == "gae"
    assert cfg.rollout.collect_prev_infos
    assert not cfg.rollout.collect_transitions
    assert "replay_buffer" not in cfg.algorithm
    assert cfg.actor.model.use_state
    assert cfg.actor.model.state_dim == 14
    assert cfg.env.train.residual.use_state
    assert cfg.env.eval.residual.use_state
    assert not cfg.env.train.residual.base_model.openpi.discrete_state_input
    assert not cfg.env.eval.residual.base_model.openpi.discrete_state_input
    assert cfg.env.train.include_states_in_obs
    cfg.actor.model.num_action_chunks = 10
    with pytest.raises(ValueError, match="num_action_chunks"):
        validate_residual_cfg(cfg)


def test_native_cnn_statistics_match_residual_channels(encoder_path):
    residual = ResidualPolicy(model_config(encoder_path)).eval()
    native = CNNPolicy(residual.cfg).eval()
    native.load_state_dict(
        {
            key: value
            for key, value in residual.state_dict().items()
            if key in native.state_dict()
        }
    )
    obs = observation()
    packed = residual._prepare_cnn_obs(obs)
    torch.manual_seed(17)
    expected_actions, expected = native.predict_action_batch(packed)
    torch.manual_seed(17)
    actual_actions, actual = residual.predict_action_batch(obs)
    torch.testing.assert_close(actual_actions[..., :6], expected_actions)
    torch.testing.assert_close(
        actual["prev_logprobs"][..., :6], expected["prev_logprobs"]
    )
    torch.testing.assert_close(actual["prev_values"], expected["prev_values"])
    residual.train()
    native.train()
    actual_output = residual(forward_inputs=actual["forward_inputs"])
    expected_output = native(forward_inputs=expected["forward_inputs"])
    torch.testing.assert_close(
        actual_output["logprobs"][..., :6], expected_output["logprobs"]
    )
    torch.testing.assert_close(
        actual_output["entropy"][:, 0, :6], expected_output["entropy"]
    )
    torch.testing.assert_close(actual_output["values"], expected_output["values"])
