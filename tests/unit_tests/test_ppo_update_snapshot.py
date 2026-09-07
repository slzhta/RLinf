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

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.utils.ppo_update_snapshot import FirstPPOUpdateSnapshot


def _actor(tmp_path, co_training=False):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "actor": {
                    "model": {"model_type": "cnn_policy"},
                    "global_batch_size": 4,
                    "debug_first_update": {"enabled": True},
                },
                "algorithm": {
                    "update_epoch": 2,
                    "sim_real_rl_co_training": co_training,
                },
                "runner": {"logger": {"log_path": str(tmp_path)}},
                "env": {"train": {"env_type": "maniskill"}},
            }
        ),
        _rank=0,
        _world_size=1,
        _use_keyed_actor_trajectory=lambda: True,
        _co_training_buffer_metrics={
            "buffer/train_real_rollouts": 1,
            "buffer/train_sim_rollouts": 3,
        },
        rollout_batch={
            "prev_logprobs": torch.zeros(2, 4, 1),
            "prev_values": torch.zeros(3, 4, 1),
            "advantages": torch.arange(8).reshape(2, 4, 1).float(),
            "forward_inputs": {
                "main_images": torch.zeros(2, 4, 8, 8, 3, dtype=torch.uint8)
            },
        },
        model=model,
        get_model_state_dict=lambda **kwargs: model.state_dict(),
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        grad_scaler=SimpleNamespace(state_dict=lambda: {}),
        optimizer_steps=0,
        critic_warmup_steps=1,
        store_requires_grad_param_name=list(dict(model.named_parameters())),
        version=0,
    )


@pytest.mark.parametrize("co_training", [False, True])
def test_snapshot_keeps_original_batch_and_training_state(tmp_path, co_training):
    actor = _actor(tmp_path, co_training)
    shuffle_id = torch.arange(7, -1, -1)
    rng = torch.get_rng_state().clone()
    snapshot = FirstPPOUpdateSnapshot(actor, shuffle_id)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert snapshot.checkpoint_steps == {1, 2, 4}
    batch = torch.load(snapshot.directory / "batch.pt", weights_only=False)
    assert batch["trajectory_domains"] == (
        ["real", "sim", "sim", "sim"] if co_training else ["sim"] * 4
    )
    torch.testing.assert_close(batch["shuffle_id"], shuffle_id)
    torch.testing.assert_close(
        batch["rollout_batch"]["advantages"], actor.rollout_batch["advantages"]
    )
    assert batch["rollout_batch"]["prev_values"].shape[0] == 3
    before = torch.load(snapshot.directory / "before_update.pt", weights_only=False)
    torch.testing.assert_close(before["model"]["weight"], actor.model.weight)
    assert before["optimizer_parameter_names"] == [["weight", "bias"]]
    assert before["optimizer_full"]["param_groups"][0]["params"] == ["weight", "bias"]
    assert before["critic_warmup_steps"] == 1
    actor.rollout_batch["advantages"].zero_()
    assert batch["rollout_batch"]["advantages"].sum() > 0
    assert actor.model.training


def test_snapshot_marks_completion_after_selected_steps(tmp_path):
    actor = _actor(tmp_path)
    snapshot = FirstPPOUpdateSnapshot(actor, torch.arange(8))
    for step in range(1, 5):
        actor.optimizer_steps = step
        snapshot.save_step(actor)
    manifest = json.loads((snapshot.directory / "manifest.json").read_text())
    assert manifest["status"] == "capturing"
    assert manifest["saved_optimizer_steps"] == [1, 2, 4]
    assert not (snapshot.directory / "after_optimizer_step_0003.pt").exists()
    snapshot.finish(actor, {"actor/lr": [1e-5]}, {"actor/lr": 1e-5})
    manifest = json.loads((snapshot.directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (snapshot.directory / "update_result.pt").is_file()
    with pytest.raises(FileExistsError):
        FirstPPOUpdateSnapshot(_actor(tmp_path), torch.arange(8))


def test_snapshot_rejects_unsupported_actor_and_invalid_domains(tmp_path):
    actor = _actor(tmp_path, co_training=True)
    actor._world_size = 2
    with pytest.raises(ValueError, match="one actor rank"):
        FirstPPOUpdateSnapshot(actor, torch.arange(8))
    actor._world_size = 1
    actor._co_training_buffer_metrics["buffer/train_sim_rollouts"] = 4
    with pytest.raises(ValueError, match="domain counts"):
        FirstPPOUpdateSnapshot(actor, torch.arange(8))
    assert not (tmp_path / "debug_first_update").exists()


def test_snapshot_cannot_mark_an_incomplete_update_complete(tmp_path):
    actor = _actor(tmp_path)
    snapshot = FirstPPOUpdateSnapshot(actor, torch.arange(8))
    with pytest.raises(RuntimeError, match="missing optimizer steps"):
        snapshot.finish(actor, {}, {})
    manifest = json.loads((snapshot.directory / "manifest.json").read_text())
    assert manifest["status"] == "capturing"
