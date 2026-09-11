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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.models.embodiment.cnn_policy.cnn_policy import (
    CNNConfig,
    CNNPolicy,
    _select_policy_extra_view_images,
)
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_abort_loss_mask
from rlinf.utils.utils import masked_mean, reshape_entropy, reshape_entropy_mask
from rlinf.workers.actor.async_ppo_fsdp_worker import (
    flatten_rollout_batch_for_train,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _make_hybrid_policy() -> CNNPolicy:
    policy = CNNPolicy.__new__(CNNPolicy)
    nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        action_dim=7,
        num_action_chunks=1,
        binary_action_temperature=1.0,
        logstd_range=(-3.0, -1.5),
        action_std_scale=[1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0],
        std_range=None,
    )
    policy._binary_action_indices = (6,)
    policy._continuous_action_indices = tuple(range(6))
    policy.action_scale = None
    return policy


def test_single_image_policy_drops_unused_extra_views():
    observations = {
        "main_images": torch.zeros(2, 128, 128, 3),
        "extra_view_images": torch.zeros(2, 4, 128, 128, 3),
    }
    assert _select_policy_extra_view_images(observations, image_num=1) is None


def test_multi_image_policy_validates_available_views():
    observations = {"extra_view_images": torch.zeros(2, 1, 8, 8, 3)}
    with pytest.raises(ValueError, match="requires 2 extra"):
        _select_policy_extra_view_images(observations, image_num=3)


def test_multi_image_policy_selects_only_configured_views():
    extra_view_images = torch.arange(2 * 3).reshape(2, 3, 1, 1, 1)
    selected = _select_policy_extra_view_images(
        {"extra_view_images": extra_view_images}, image_num=2
    )
    assert torch.equal(selected, extra_view_images[:, :1])


def test_hybrid_action_generation_matches_training_statistics():
    policy = _make_hybrid_policy()
    action_mean = torch.zeros(4, 7)
    action_logstd = torch.full((4, 7), -2.0)
    feature = torch.zeros(4, 8)
    policy._actor_forward_from_processed_tensors = lambda **_: (
        feature,
        feature,
        action_mean,
        action_logstd,
    )

    action, _, rollout_logprobs, _, _ = policy._generate_actions(
        states=None,
        main_images=torch.zeros(4, 8, 8, 3),
        extra_view_images=None,
        calculate_values=False,
        mode="train",
    )
    recomputed_logprobs, entropy = policy._hybrid_action_statistics(
        action_mean, policy._action_std(action_logstd), action
    )

    assert torch.all(action[:, :6].abs() < 1.0)
    assert set(action[:, 6].tolist()) <= {-1.0, 1.0}
    torch.testing.assert_close(rollout_logprobs, recomputed_logprobs)
    assert torch.isfinite(entropy).all()


def test_hybrid_eval_action_is_deterministic_and_binary():
    policy = _make_hybrid_policy()
    action_mean = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.0, 0.0, -0.01]])
    action_logstd = torch.full((1, 7), -2.0)
    feature = torch.zeros(1, 8)
    policy._actor_forward_from_processed_tensors = lambda **_: (
        feature,
        feature,
        action_mean,
        action_logstd,
    )

    first, _, _, _, _ = policy._generate_actions(
        states=None,
        main_images=torch.zeros(1, 8, 8, 3),
        extra_view_images=None,
        calculate_values=False,
        mode="eval",
    )
    second, _, _, _, _ = policy._generate_actions(
        states=None,
        main_images=torch.zeros(1, 8, 8, 3),
        extra_view_images=None,
        calculate_values=False,
        mode="eval",
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :6], torch.tanh(action_mean[:, :6]))
    assert first[0, 6].item() == -1.0


def test_binary_actions_reject_global_action_scale():
    cfg = CNNConfig(action_dim=7, binary_action_indices=[6])
    cfg.action_scale = [1.0] * 7
    with pytest.raises(ValueError, match="Binary action channels"):
        CNNPolicy(cfg)


def test_env_output_merge_fills_missing_abort_flags():
    first = EnvOutput(
        obs={"main_images": torch.zeros(1, 2, 2, 3)},
        dones=torch.zeros(1, 1, dtype=torch.bool),
        abort_flags=torch.ones(1, 1, dtype=torch.bool),
    ).to_dict()
    second = EnvOutput(
        obs={"main_images": torch.zeros(2, 2, 2, 3)},
        dones=torch.zeros(2, 1, dtype=torch.bool),
    ).to_dict()

    merged = EnvOutput.merge_env_outputs([first, second])
    assert merged["abort_flags"].tolist() == [[True], [False], [False]]


def test_abort_mask_removes_only_the_aborted_episode():
    dones = torch.zeros(5, 2, 1, dtype=torch.bool)
    aborts = torch.zeros_like(dones)
    dones[2, 0, 0] = True
    aborts[2, 0, 0] = True
    dones[3, 1, 0] = True

    loss_mask, loss_mask_sum = compute_abort_loss_mask(dones, aborts)

    assert loss_mask[:, 0, 0].tolist() == [False, False, True, True]
    assert loss_mask[:, 1, 0].tolist() == [True, True, True, True]
    assert loss_mask_sum[:, 0, 0].unique().item() == 2
    assert loss_mask_sum[:, 1, 0].unique().item() == 4


def test_co_training_concat_drops_domain_specific_nested_fields():
    real_batch = {
        "prev_logprobs": torch.zeros(2, 1, 1),
        "forward_inputs": {
            "action": torch.zeros(2, 1, 1),
            "main_images": torch.zeros(2, 1, 2, 2, 3),
            "extra_view_images": torch.zeros(2, 1, 1, 2, 2, 3),
        },
    }
    sim_batch = {
        "prev_logprobs": torch.ones(2, 1, 1),
        "forward_inputs": {
            "action": torch.ones(2, 1, 1),
            "main_images": torch.ones(2, 1, 2, 2, 3),
        },
    }
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)

    merged = actor._concat_rollout_batches_along_batch_dim([real_batch, sim_batch])

    assert merged["prev_logprobs"].shape == (2, 2, 1)
    assert merged["forward_inputs"]["action"].shape == (2, 2, 1)
    assert "extra_view_images" not in merged["forward_inputs"]


def test_flatten_reports_misaligned_nested_field():
    rollout_batch = {
        "prev_logprobs": torch.zeros(2, 2, 1),
        "forward_inputs": {
            "extra_view_images": torch.zeros(2, 1, 1, 2, 2, 3)
        },
    }

    with pytest.raises(
        ValueError,
        match=r"rollout_batch\.forward_inputs\.extra_view_images.*2.*4",
    ):
        flatten_rollout_batch_for_train(rollout_batch, torch.arange(4))


def test_chunk_entropy_and_mask_have_identical_batch_shape():
    entropy = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    mask = torch.tensor([[True], [False]])

    reshaped_entropy = reshape_entropy(entropy, "chunk_level", 7, 2)
    reshaped_mask = reshape_entropy_mask(mask, "chunk_level", 2)

    assert reshaped_entropy.shape == reshaped_mask.shape == (2,)
    assert masked_mean(reshaped_entropy, reshaped_mask).item() == 21.0


def test_tensorboard_resume_purges_only_rewritten_steps():
    assert (
        MetricLogger._get_tensorboard_purge_step("/tmp/checkpoints/global_step_40")
        == 41
    )
    assert MetricLogger._get_tensorboard_purge_step("auto") is None
