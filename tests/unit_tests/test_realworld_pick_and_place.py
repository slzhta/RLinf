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

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
    HumanPnPRewardDoneWrapper,
)
from rlinf.envs.realworld.franka.tasks.co_training_base_env import (
    FrankaCoTrainingBaseEnv,
)
from rlinf.envs.realworld.franka.tasks.pick_and_place_env import (
    FrankaPickAndPlaceConfig,
    FrankaPickAndPlaceEnv,
)


class _EventSource:
    def __init__(self):
        self.events = []

    def get_key(self):
        return None

    def get_key_event(self):
        return self.events.pop(0) if self.events else None

    def clear_key_events(self):
        self.events.clear()


class _FeedbackEnv(gym.Env):
    def __init__(self):
        self.next_truncated = False
        self.next_info = {}

    def reset(self, *, seed=None, options=None):
        return {"observation": 0}, {}

    def step(self, action):
        return (
            {"observation": action},
            123.0,
            False,
            self.next_truncated,
            self.next_info,
        )


def _make_feedback_wrapper():
    listener = _EventSource()
    wrapper = HumanPnPRewardDoneWrapper(
        _FeedbackEnv(), listener=listener, wait_for_reset_ready=False
    )
    wrapper.reset()
    return wrapper, listener


def test_human_success_is_a_trainable_terminal_reward():
    wrapper, listener = _make_feedback_wrapper()
    listener.events.append("s")
    _, reward, terminated, truncated, info = wrapper.step(0)

    assert reward == 1.0
    assert terminated and not truncated
    assert info["success"]
    assert not info["discard_trajectory"]


def test_human_abort_is_discarded_without_becoming_failure():
    wrapper, listener = _make_feedback_wrapper()
    listener.events.append("x")
    _, reward, terminated, truncated, info = wrapper.step(0)

    assert reward == 0.0
    assert not terminated and truncated
    assert not info["fail"]
    assert info["discard_trajectory"]


def test_timeout_and_collision_are_trainable_failures():
    wrapper, _ = _make_feedback_wrapper()
    wrapper.env.next_truncated = True
    _, _, terminated, truncated, info = wrapper.step(0)
    assert terminated and not truncated
    assert info["timeout_failure"] and info["fail"]
    assert not info["discard_trajectory"]

    wrapper, _ = _make_feedback_wrapper()
    wrapper.env.next_info = {"safety_collision_failure": True}
    _, _, terminated, truncated, info = wrapper.step(0)
    assert terminated and not truncated
    assert info["collision_failure"] and info["fail"]
    assert not info["discard_trajectory"]


def test_contact_detection_is_opt_in_and_ignores_closed_gripper():
    env = FrankaPickAndPlaceEnv.__new__(FrankaPickAndPlaceEnv)
    env.config = SimpleNamespace(
        enable_downward_contact_failure=True,
        contact_downward_action_threshold=0.05,
        contact_force_threshold=10.0,
    )
    env._franka_state = SimpleNamespace(gripper_open=False)
    downward_action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])

    assert not env._is_downward_contact(downward_action, 20.0)
    env._franka_state.gripper_open = True
    assert env._is_downward_contact(downward_action, 20.0)
    env.config.enable_downward_contact_failure = False
    assert not env._is_downward_contact(downward_action, 20.0)


def test_downward_contact_retreat_is_terminal_but_not_discarded(monkeypatch):
    env = object.__new__(FrankaPickAndPlaceEnv)
    env.config = FrankaPickAndPlaceConfig(
        camera_serials=["dummy"],
        is_dummy=True,
        enable_downward_contact_failure=True,
        contact_force_threshold=10.0,
        contact_downward_action_threshold=0.05,
        contact_retreat_distance=0.05,
        contact_retreat_timeout=0.5,
    )
    env._contact_force_baseline = np.zeros(3)
    env._franka_state = SimpleNamespace(
        tcp_force=np.array([0.0, 0.0, 12.0]),
        tcp_pose=np.array([0.5, 0.1, 0.05, 0.0, 0.0, 0.0, 1.0]),
        gripper_open=True,
    )
    env._target_pose = np.array([0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0])
    env._logger = type(
        "Logger", (), {"warning": staticmethod(lambda *args, **kwargs: None)}
    )()
    sent_poses = []

    def fake_base_step(self, action):
        return {"state": "before_retreat"}, 0.0, False, False, {}

    def fake_move_action(pose):
        sent_poses.append(np.asarray(pose).copy())

    def fake_interpolate_move(pose, timeout):
        assert timeout == 0.5
        env._franka_state.tcp_pose = np.asarray(pose).copy()

    monkeypatch.setattr(FrankaCoTrainingBaseEnv, "step", fake_base_step)
    monkeypatch.setattr(env, "_move_action", fake_move_action)
    monkeypatch.setattr(env, "_interpolate_move", fake_interpolate_move)
    monkeypatch.setattr(env, "_clip_position_to_safety_box", lambda pose: pose)
    monkeypatch.setattr(env, "_get_observation", lambda: {"state": "retreated"})

    obs, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    )

    assert obs == {"state": "retreated"}
    assert reward == 0.0
    assert terminated and not truncated
    assert info["safety_collision_failure"]
    assert not info["discard_trajectory"]
    assert np.allclose(sent_poses[0], [0.5, 0.1, 0.05, 0.0, 0.0, 0.0, 1.0])
    assert np.isclose(env._target_pose[2], 0.10)


def test_pnp_downward_motion_uses_single_base_target(monkeypatch):
    env = object.__new__(FrankaPickAndPlaceEnv)
    sent_poses = []
    monkeypatch.setattr(
        env, "_move_action", lambda pose: sent_poses.append(pose.copy())
    )

    target = np.array([0.5, 0.1, 0.04, 0.0, 0.0, 0.0, 1.0])
    action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    env._execute_arm_motion(target, action)

    assert len(sent_poses) == 1
    assert np.array_equal(sent_poses[0], target)


def test_below_threshold_contact_check_does_not_poll_controller(monkeypatch):
    env = object.__new__(FrankaPickAndPlaceEnv)
    env.config = FrankaPickAndPlaceConfig(
        camera_serials=["dummy"],
        is_dummy=False,
        enable_downward_contact_failure=True,
        contact_force_threshold=10.0,
        contact_force_confirmation_samples=3,
        contact_monitor_interval=0.02,
    )
    env._franka_state = SimpleNamespace(gripper_open=True)
    env._controller = SimpleNamespace(
        get_state=lambda: (_ for _ in ()).throw(
            AssertionError("below-threshold checks must not poll the controller")
        )
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.tasks.pick_and_place_env.time.sleep",
        lambda _: (_ for _ in ()).throw(
            AssertionError("below-threshold checks must not sleep")
        ),
    )

    action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    assert env._confirm_downward_contact(action, 9.99) is None


def test_downward_contact_requires_consecutive_force_samples(monkeypatch):
    class _Result:
        def __init__(self, value=None):
            self.value = value

        def wait(self):
            return self.value

    class _Controller:
        def __init__(self, states):
            self.states = iter(states)

        def get_state(self):
            return _Result([next(self.states)])

    action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
    pose = np.array([0.5, 0.1, 0.05, 0.0, 0.0, 0.0, 1.0])
    env = object.__new__(FrankaPickAndPlaceEnv)
    env.config = FrankaPickAndPlaceConfig(
        camera_serials=["dummy"],
        is_dummy=False,
        enable_downward_contact_failure=True,
        contact_force_threshold=10.0,
        contact_force_confirmation_samples=3,
        contact_monitor_interval=0.02,
    )
    env._contact_force_baseline = np.zeros(3)
    env._franka_state = SimpleNamespace(
        tcp_force=np.array([0.0, 0.0, 15.74]),
        tcp_pose=pose.copy(),
        gripper_open=True,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.tasks.pick_and_place_env.time.sleep",
        lambda _: None,
    )

    env._controller = _Controller(
        [
            SimpleNamespace(
                tcp_force=np.array([0.0, 0.0, 4.0]),
                tcp_pose=pose.copy(),
                gripper_open=True,
            )
        ]
    )
    assert env._confirm_downward_contact(action, 15.74) is None

    env._controller = _Controller(
        [
            SimpleNamespace(
                tcp_force=np.array([0.0, 0.0, 11.0]),
                tcp_pose=pose.copy(),
                gripper_open=True,
            ),
            SimpleNamespace(
                tcp_force=np.array([0.0, 0.0, 12.0]),
                tcp_pose=pose.copy(),
                gripper_open=True,
            ),
        ]
    )
    assert env._confirm_downward_contact(action, 10.5) == 12.0


def test_gripper_command_interval_suppresses_rapid_reversal(monkeypatch):
    class _Result:
        def wait(self):
            return None

    class _Controller:
        def __init__(self):
            self.commands = []

        def close_gripper(self):
            self.commands.append("close")
            return _Result()

        def open_gripper(self):
            self.commands.append("open")
            return _Result()

    env = object.__new__(FrankaPickAndPlaceEnv)
    env.config = SimpleNamespace(
        use_zero_one_gripper_action=False,
        binary_gripper_threshold=0.5,
        gripper_min_command_interval=1.0,
    )
    env._controller = _Controller()
    env._franka_state = SimpleNamespace(gripper_open=True)
    env._last_gripper_command_time = float("-inf")
    times = iter([10.0, 10.5, 11.1])
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.franka_env.time.monotonic",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.franka_env.time.sleep", lambda _: None
    )

    assert env._gripper_action(-1.0)
    env._franka_state.gripper_open = False
    assert not env._gripper_action(1.0)
    assert env._gripper_action(1.0)
    assert env._controller.commands == ["close", "open"]
