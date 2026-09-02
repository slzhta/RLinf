from collections import deque

import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    HumanPnPRewardDoneWrapper,
)
from rlinf.envs.realworld.franka.tasks.pick_and_place_env import (
    FrankaPickAndPlaceEnv,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers.record_video import RecordVideo
from rlinf.utils.metric_utils import compute_abort_loss_mask


class _FakePnPEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
    observation_space = gym.spaces.Dict(
        {"marker": gym.spaces.Box(0.0, 10.0, shape=(1,), dtype=np.float32)}
    )

    def __init__(self, *, terminated: bool = False, truncated: bool = False):
        self.terminated = terminated
        self.truncated = truncated
        self.refresh_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"marker": np.array([0.0], dtype=np.float32)}, {}

    def step(self, action):
        return (
            {"marker": np.array([1.0], dtype=np.float32)},
            0.75,
            self.terminated,
            self.truncated,
            {},
        )

    def refresh_observation(self):
        self.refresh_count += 1
        return {"marker": np.array([2.0], dtype=np.float32)}


class _EventListener:
    def __init__(self, *events: str):
        self.events = deque(events)

    def get_key(self):
        return None

    def get_key_event(self):
        return self.events.popleft() if self.events else None

    def clear_key_events(self):
        pass


class _HeldKeyListener:
    def __init__(self, key: str):
        self.key = key

    def get_key(self):
        return self.key


def _step_with_feedback(feedback_key: str, **env_kwargs):
    wrapper = HumanPnPRewardDoneWrapper(
        _FakePnPEnv(**env_kwargs),
        listener=_EventListener(feedback_key),
        wait_for_reset_ready=False,
    )
    return wrapper.step(np.zeros(7, dtype=np.float32))


def test_human_pnp_terminal_feedback_semantics():
    _, reward, terminated, truncated, info = _step_with_feedback("s")
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["human_feedback"] == "success"
    assert info["human_feedback_should_train"]

    _, reward, terminated, truncated, info = _step_with_feedback("f")
    assert reward == 0.0
    assert terminated
    assert not truncated
    assert info["human_feedback"] == "failure"
    assert info["human_feedback_should_train"]

    _, reward, terminated, truncated, info = _step_with_feedback("x")
    assert reward == 0.0
    assert not terminated
    assert truncated
    assert info["human_feedback"] == "abort"
    assert info["discard_trajectory"]


def test_human_pnp_wrapper_preserves_hardware_termination():
    _, reward, terminated, truncated, _ = _step_with_feedback(
        "unmapped", terminated=True, truncated=True
    )
    assert reward == 0.0
    assert terminated
    assert truncated


def test_human_pnp_reset_waits_for_ready_and_refreshes_observation():
    env = _FakePnPEnv()
    wrapper = HumanPnPRewardDoneWrapper(
        env,
        listener=_EventListener("r"),
        reset_poll_interval=0.001,
    )

    observation, info = wrapper.reset()

    np.testing.assert_array_equal(observation["marker"], np.array([2.0]))
    assert env.refresh_count == 1
    assert info["human_reset_ready"]


def test_human_pnp_debounces_a_held_feedback_key_across_reset():
    wrapper = HumanPnPRewardDoneWrapper(
        _FakePnPEnv(),
        listener=_HeldKeyListener("s"),
        wait_for_reset_ready=False,
    )
    wrapper.reset()

    _, reward, terminated, _, info = wrapper.step(np.zeros(7, dtype=np.float32))

    assert reward == 0.0
    assert not terminated
    assert not info["human_feedback_received"]


def test_abort_loss_mask_removes_only_aborted_episodes():
    dones = torch.zeros((7, 2, 1), dtype=torch.bool)
    aborts = torch.zeros_like(dones)

    dones[3, 0, 0] = True
    aborts[3, 0, 0] = True
    dones[6, 0, 0] = True

    dones[2, 1, 0] = True
    dones[5, 1, 0] = True
    aborts[5, 1, 0] = True

    loss_mask, loss_mask_sum = compute_abort_loss_mask(dones, aborts)

    expected = torch.tensor(
        [
            [False, True],
            [False, True],
            [False, False],
            [True, False],
            [True, False],
            [True, True],
        ],
        dtype=torch.bool,
    ).unsqueeze(-1)
    assert torch.equal(loss_mask, expected)
    assert torch.equal(loss_mask_sum[:, 0], torch.full((6, 1), 3))
    assert torch.equal(loss_mask_sum[:, 1], torch.full((6, 1), 3))


def test_realworld_observation_uses_explicit_pnp_state_order():
    env = object.__new__(RealWorldEnv)
    env.state_keys = ["arm_joint_position", "tcp_pose", "gripper_open_state"]
    env.include_states_in_obs = True
    env.main_image_key = "wrist_1"
    env.task_descriptions = ["pnp"]
    raw_observation = {
        "state": {
            "tcp_pose": np.arange(6, dtype=np.float32)[None, :] + 10,
            "gripper_open_state": np.array([[1.0]], dtype=np.float32),
            "arm_joint_position": np.arange(7, dtype=np.float32)[None, :],
        },
        "frames": {
            "wrist_2": np.zeros((1, 128, 128, 3), dtype=np.uint8),
            "wrist_1": np.ones((1, 128, 128, 3), dtype=np.uint8),
        },
    }

    observation = RealWorldEnv._wrap_obs(env, raw_observation)

    expected_state = np.concatenate([np.arange(7), np.arange(6) + 10, np.array([1.0])])
    np.testing.assert_array_equal(observation["states"].numpy()[0], expected_state)
    assert observation["states"].shape == (1, 14)
    assert observation["main_images"].shape == (1, 128, 128, 3)
    assert observation["extra_view_images"].shape == (1, 1, 128, 128, 3)


def test_pnp_video_tiles_main_and_auxiliary_camera_views():
    env = _FakePnPEnv()
    env.seed = 0
    recorder = RecordVideo(env, {"include_extra_views": True})
    observation = {
        "main_images": np.ones((1, 8, 8, 3), dtype=np.uint8),
        "extra_view_images": np.zeros((1, 1, 8, 8, 3), dtype=np.uint8),
    }

    frame_batches = recorder._extract_frame_batches(observation)
    recorder.close()

    assert len(frame_batches) == 1
    assert len(frame_batches[0]) == 1
    assert frame_batches[0][0].shape == (8, 16, 3)
    assert np.all(frame_batches[0][0][:, :8] == 1)
    assert np.all(frame_batches[0][0][:, 8:] == 0)


def test_realworld_pnp_dummy_env_has_aligned_spaces_and_zero_task_reward():
    env = FrankaPickAndPlaceEnv(
        override_cfg={
            "camera_serials": ["main", "aux"],
            "is_dummy": True,
            "target_ee_pose": [0.5, 0.0, 0.0, 3.14, 0.0, 0.0],
        }
    )

    wrapped_env = Quat2EulerWrapper(env)
    observation, _ = wrapped_env.reset()

    assert env.action_space.shape == (7,)
    assert list(observation["state"]) == [
        "arm_joint_position",
        "tcp_pose",
        "gripper_open_state",
    ]
    assert observation["state"]["gripper_open_state"].shape == (1,)
    assert observation["state"]["tcp_pose"].shape == (6,)
    assert observation["state"]["tcp_pose"].dtype == np.float32
    assert env._calc_step_reward(observation) == 0.0
