from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.envs.maniskill.tasks.digital_twin.pick_and_place import (
    PickAndPlaceDigitalTwinEnv,
)
from rlinf.workers.env.env_worker import EnvWorker


class _Agent:
    def __init__(self, grasped: torch.Tensor):
        self.grasped = grasped

    def is_grasping(self, _actor) -> torch.Tensor:
        return self.grasped


def _make_task(num_envs: int = 1) -> PickAndPlaceDigitalTwinEnv:
    task = object.__new__(PickAndPlaceDigitalTwinEnv)
    task.num_envs = num_envs
    task.device = torch.device("cpu")
    task.task_alignment = {
        "success_stability_steps": 3,
        "success_max_linear_speed": 0.05,
        "success_max_angular_speed": 0.5,
        "sparse_grasp_reward": 0.1,
        "sparse_lift_reward": 0.2,
        "sparse_place_reward": 0.3,
        "sparse_success_reward": 1.0,
        "sparse_drop_penalty": -0.2,
    }
    task._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)

    target_position = torch.tensor([[-0.07, -0.055, 0.0075]]).repeat(num_envs, 1)
    source_position = torch.tensor([[-0.07, 0.125, 0.0075]]).repeat(num_envs, 1)
    goal_position = target_position.clone()
    goal_position[:, 2] += task._cube_center_local_z()
    task.target_tray = SimpleNamespace(pose=SimpleNamespace(p=target_position))
    task.source_tray = SimpleNamespace(pose=SimpleNamespace(p=source_position))
    task.cube = SimpleNamespace(
        pose=SimpleNamespace(p=goal_position),
        linear_velocity=torch.zeros(num_envs, 3),
        angular_velocity=torch.zeros(num_envs, 3),
    )
    task.agent = _Agent(torch.zeros(num_envs, dtype=torch.bool))
    task._reset_episode_metric_state()
    task._reset_sparse_reward_state()
    return task


def test_success_requires_distinct_stable_control_steps():
    task = _make_task()

    for step in (1, 2):
        task._elapsed_steps.fill_(step)
        info = task.evaluate()
        assert not info["success"].item()
        assert info["success_stable_steps"].item() == step

        repeated_info = task.evaluate()
        assert repeated_info["success_stable_steps"].item() == step

    task._elapsed_steps.fill_(3)
    info = task.evaluate()
    assert info["success"].item()
    assert info["success_stable_steps"].item() == 3

    task.cube.linear_velocity[:, 0] = 1.0
    task._elapsed_steps.fill_(4)
    assert task.evaluate()["success"].item()

    task._reset_episode_metric_state(torch.tensor([0]))
    assert not task._episode_success_once.item()
    assert task._success_stable_steps.item() == 0


def test_unrecoverable_object_terminates_as_failure():
    task = _make_task()
    task.cube.pose.p[:, 2] = -0.03
    task._elapsed_steps.fill_(1)

    info = task.evaluate()

    assert info["fail"].item()
    assert info["is_obj_unrecoverable"].item()
    assert not info["success"].item()


def test_sparse_success_path_has_consistent_milestone_return():
    task = _make_task()
    common = {
        "cube_to_goal_dist": torch.zeros(1),
        "is_obj_placed": torch.tensor([False]),
        "success": torch.tensor([False]),
    }
    grasp_reward = task._compute_sparse_milestone_reward(
        {
            **common,
            "is_grasped": torch.tensor([True]),
            "is_obj_lifted": torch.tensor([False]),
        }
    )
    lift_reward = task._compute_sparse_milestone_reward(
        {
            **common,
            "is_grasped": torch.tensor([True]),
            "is_obj_lifted": torch.tensor([True]),
        }
    )
    place_and_success_reward = task._compute_sparse_milestone_reward(
        {
            **common,
            "is_grasped": torch.tensor([False]),
            "is_obj_lifted": torch.tensor([False]),
            "is_obj_placed": torch.tensor([True]),
            "success": torch.tensor([True]),
        }
    )

    total_reward = grasp_reward + lift_reward + place_and_success_reward
    torch.testing.assert_close(total_reward, torch.tensor([1.6]))
    torch.testing.assert_close(place_and_success_reward, torch.tensor([1.3]))


def test_maniskill_reset_respects_explicit_seed():
    class _RawEnv:
        def __init__(self):
            self.calls = []

        def reset(self, *, seed, options):
            self.calls.append((seed, options))
            return {"obs": torch.zeros(1)}, {}

    env = object.__new__(ManiskillEnv)
    env.seed = 7
    env.use_fixed_reset_state_ids = False
    env.env = _RawEnv()
    env._show_goal_site_visual = lambda: None
    env._wrap_obs = lambda raw_obs, infos: raw_obs
    env._reset_metrics = lambda _env_idx=None: None

    env.reset(seed=11)
    env.reset()

    assert env.env.calls == [(11, {}), (7, {})]


class _EvalEnv:
    def __init__(self):
        self.step_index = 0
        self.actions = []
        self.done_sequence = (
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            torch.tensor([True, True]),
        )

    def chunk_step(self, chunk_actions):
        self.actions.append(
            chunk_actions.clone()
            if isinstance(chunk_actions, torch.Tensor)
            else np.array(chunk_actions, copy=True)
        )
        dones = self.done_sequence[self.step_index]
        self.step_index += 1
        obs = {"states": torch.zeros(2, 1)}
        infos = {
            "episode": {
                "success_once": torch.tensor([True, False]),
                "episode_len": torch.tensor([10, 20]),
            }
        }
        return (
            [obs],
            torch.zeros(2, 1),
            dones[:, None],
            torch.zeros(2, 1, dtype=torch.bool),
            [infos],
        )


@pytest.mark.parametrize("action_container", ["tensor", "array"])
def test_evaluation_records_each_environment_once(monkeypatch, action_container):
    def _prepare_actions(**kwargs):
        actions = kwargs["raw_chunk_actions"]
        if action_container == "tensor":
            return actions.clone()
        return actions.cpu().numpy().copy()

    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions", _prepare_actions
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"eval": {"env_type": "maniskill", "auto_reset": False}},
            "actor": {
                "model": {
                    "model_type": "cnn_policy",
                    "num_action_chunks": 1,
                    "action_dim": 7,
                    "policy_setup": "panda-ee-dpos",
                }
            },
        }
    )
    worker.eval_env_list = [_EvalEnv()]
    worker.eval_finished = [torch.zeros(2, dtype=torch.bool)]
    worker.use_external_reward_model = False
    actions = torch.ones(2, 1, 7)

    _, first_info = worker.env_evaluate_step(actions, stage_id=0)
    _, second_info = worker.env_evaluate_step(actions, stage_id=0)
    _, repeated_info = worker.env_evaluate_step(actions, stage_id=0)

    assert first_info["episode_len"].tolist() == [10]
    assert second_info["episode_len"].tolist() == [20]
    assert repeated_info == {}
    assert worker.eval_finished[0].tolist() == [True, True]
    recorded_actions = [
        torch.as_tensor(value) for value in worker.eval_env_list[0].actions
    ]
    torch.testing.assert_close(recorded_actions[0], torch.ones_like(actions))
    torch.testing.assert_close(recorded_actions[1][0], torch.zeros_like(actions[0]))
    torch.testing.assert_close(recorded_actions[1][1], torch.ones_like(actions[1]))
    torch.testing.assert_close(recorded_actions[2], torch.zeros_like(actions))


def test_auto_reset_evaluation_records_later_episodes(monkeypatch):
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda **kwargs: kwargs["raw_chunk_actions"],
    )
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {"eval": {"env_type": "maniskill", "auto_reset": True}},
            "actor": {
                "model": {
                    "model_type": "cnn_policy",
                    "num_action_chunks": 1,
                    "action_dim": 7,
                    "policy_setup": "panda-ee-dpos",
                }
            },
        }
    )
    worker.eval_env_list = [_EvalEnv()]
    worker.eval_finished = [torch.zeros(2, dtype=torch.bool)]
    worker.use_external_reward_model = False
    actions = torch.ones(2, 1, 7)

    _, first_info = worker.env_evaluate_step(actions, stage_id=0)
    _, second_info = worker.env_evaluate_step(actions, stage_id=0)
    _, third_info = worker.env_evaluate_step(actions, stage_id=0)

    assert first_info["episode_len"].tolist() == [10]
    assert second_info["episode_len"].tolist() == [20]
    assert third_info["episode_len"].tolist() == [10]
    for recorded in worker.eval_env_list[0].actions:
        torch.testing.assert_close(torch.as_tensor(recorded), actions)


def test_maniskill_eval_batches_use_deterministic_non_overlapping_seeds():
    class _ResetEnv:
        def __init__(self):
            self.seeds = []

        def reset(self, **kwargs):
            self.seeds.append(kwargs.get("seed"))
            return {}, {}

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "eval": {
                    "env_type": "maniskill",
                    "auto_reset": False,
                    "use_fixed_reset_state_ids": False,
                    "seed": 50000,
                }
            }
        }
    )
    worker.eval_env_list = [_ResetEnv()]
    worker.eval_num_envs_per_stage = 64
    worker.stage_num = 1
    worker._world_size = 1
    worker._rank = 0

    worker._reset_eval_env(stage_id=0, eval_rollout_epoch=0)
    worker._reset_eval_env(stage_id=0, eval_rollout_epoch=1)

    assert worker.eval_env_list[0].seeds == [50000, 50064]
