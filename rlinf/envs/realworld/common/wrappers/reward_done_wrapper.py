# Copyright 2025 The RLinf Authors.
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

import time
from typing import Any, Protocol, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.utils.logging import get_logger


class KeyEventSource(Protocol):
    """Minimal keyboard interface used by human-feedback wrappers."""

    def get_key(self) -> str | None: ...


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        assert self.reward_mode in ["always_replace"]

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        return observation, reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        return last_intervened, keyboard_reward, terminated


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
            return last_intervened, done, reward

        last_intervened = True
        if key == "a":
            reward = -1
            done = True
            last_intervened = True
        elif key == "b":
            reward = 0
            last_intervened = True
        elif key == "c":
            reward = 1
            done = True
            last_intervened = True
        return last_intervened, done, reward


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 0.1, 1]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2

        if self.reward_stage == 2:
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = False
        return last_intervened, done, reward


class HumanPnPRewardDoneWrapper(gym.Wrapper):
    """Provide terminal-only human rewards and manual reset for real PnP.

    Feedback keys are ``S`` for success, ``F`` for unrecoverable failure,
    ``X`` for an administrative or safety abort, and ``R`` when the workspace
    is ready for the next episode. Key matching is case-insensitive.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        listener: KeyEventSource | None = None,
        success_key: str = "s",
        failure_key: str = "f",
        abort_key: str = "x",
        ready_key: str = "r",
        wait_for_reset_ready: bool = True,
        reset_poll_interval: float = 0.05,
        reset_ready_timeout: float | None = None,
    ):
        super().__init__(env)
        keys = [success_key, failure_key, abort_key, ready_key]
        normalized_keys = [key.lower() for key in keys]
        if any(len(key) != 1 for key in normalized_keys):
            raise ValueError("Human feedback keys must be single characters.")
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("Human feedback keys must be unique.")
        if reset_poll_interval <= 0:
            raise ValueError("reset_poll_interval must be positive.")
        if reset_ready_timeout is not None and reset_ready_timeout <= 0:
            raise ValueError("reset_ready_timeout must be positive when set.")

        (
            self.success_key,
            self.failure_key,
            self.abort_key,
            self.ready_key,
        ) = normalized_keys
        self.listener = listener or KeyboardListener()
        self.wait_for_reset_ready = wait_for_reset_ready
        self.reset_poll_interval = reset_poll_interval
        self.reset_ready_timeout = reset_ready_timeout
        self._logger = get_logger()
        self._last_polled_key: str | None = None
        self._episode_id = -1
        self._step_id = 0
        self._terminal_feedback_received = False

    def _get_key_event(self) -> str | None:
        event_reader = getattr(self.listener, "get_key_event", None)
        if event_reader is not None:
            key = event_reader()
            return key.lower() if key is not None else None

        key = self.listener.get_key()
        if key is None:
            self._last_polled_key = None
            return None
        key = key.lower()
        if key == self._last_polled_key:
            return None
        self._last_polled_key = key
        return key

    def _clear_key_events(self) -> None:
        clear_events = getattr(self.listener, "clear_key_events", None)
        if clear_events is not None:
            clear_events()
        self._last_polled_key = self.listener.get_key()

    def _wait_until_ready(self) -> tuple[float, float]:
        self._clear_key_events()
        wait_started_at = time.time()
        deadline = (
            time.monotonic() + self.reset_ready_timeout
            if self.reset_ready_timeout is not None
            else None
        )
        self._logger.info(
            "PnP robot is at rest. Reset the object, leave the workspace, and press R."
        )
        while True:
            if self._get_key_event() == self.ready_key:
                ready_at = time.time()
                self._logger.info("PnP reset-ready feedback accepted.")
                return ready_at, ready_at - wait_started_at
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out while waiting for PnP reset-ready feedback."
                )
            time.sleep(self.reset_poll_interval)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        reset_ready_at = time.time()
        reset_wait_seconds = 0.0
        if self.wait_for_reset_ready:
            reset_ready_at, reset_wait_seconds = self._wait_until_ready()
            refresh_observation = getattr(
                self.env.unwrapped, "refresh_observation", None
            )
            if refresh_observation is None:
                raise RuntimeError(
                    "HumanPnPRewardDoneWrapper requires refresh_observation() "
                    "when wait_for_reset_ready=True."
                )
            observation = refresh_observation()
        else:
            self._clear_key_events()

        self._episode_id += 1
        self._step_id = 0
        self._terminal_feedback_received = False
        info.update(
            {
                "human_reset_ready": True,
                "human_reset_ready_timestamp": reset_ready_at,
                "human_reset_wait_seconds": reset_wait_seconds,
            }
        )
        return observation, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        step_started_at = time.time()
        observation, _, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward = 0.0
        feedback = None
        should_train = True

        key = self._get_key_event()
        if not self._terminal_feedback_received:
            if key == self.success_key:
                feedback = "success"
                reward = 1.0
                terminated = True
            elif key == self.failure_key:
                feedback = "failure"
                terminated = True
            elif key == self.abort_key:
                feedback = "abort"
                truncated = True
                should_train = False

        if feedback is not None:
            self._terminal_feedback_received = True
            feedback_at = time.time()
            self._logger.info(
                "PnP human feedback accepted: episode=%d step=%d feedback=%s",
                self._episode_id,
                self._step_id,
                feedback,
            )
        else:
            feedback_at = 0.0

        info.update(
            {
                "human_feedback": feedback or "none",
                "human_feedback_received": feedback is not None,
                "human_feedback_episode_id": self._episode_id,
                "human_feedback_step_id": self._step_id,
                "human_feedback_timestamp": feedback_at,
                "human_feedback_latency_seconds": (
                    feedback_at - step_started_at if feedback is not None else 0.0
                ),
                "human_feedback_reward": reward,
                "human_feedback_should_train": should_train,
                "discard_trajectory": not should_train,
            }
        )
        self._step_id += 1
        return observation, reward, bool(terminated), bool(truncated), info
