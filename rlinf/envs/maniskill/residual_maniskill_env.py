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
import weakref
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.envs.residual import BaseActionCache
from rlinf.models.embodiment.residual_policy.validation import (
    validate_success_only_reward_cfg,
)

# Train/eval environments in the same worker share one frozen base, not its cache.
_BASE_MODELS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


class ResidualManiskillEnv(ManiskillEnv):
    """Expose normalized action residuals as the actions of a ManiSkill MDP.

    The existing env worker handles trajectories and terminal observations.
    OpenPI sees the original full-resolution observations including states;
    the existing NHWC images and base plan are sent to rollout/training.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Any,
    ) -> None:
        validate_success_only_reward_cfg(cfg)
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        self.base_policy = None
        self.base_cache = BaseActionCache(
            num_envs, cfg.residual.base_horizon, cfg.residual.action_dim, self.device
        )
        self.residual_scale = torch.tensor(
            list(cfg.residual.action_scale), device=self.device, dtype=torch.float32
        )
        if (
            self.residual_scale.shape != (cfg.residual.action_dim,)
            or not torch.isfinite(self.residual_scale).all()
            or (self.residual_scale < 0).any()
        ):
            raise ValueError(
                "Residual scale must be finite, nonnegative, and match action_dim."
            )

    def _load_base(self) -> None:
        if self.base_policy is not None:
            return
        from rlinf.models.embodiment.openpi import get_model

        model_cfg = self.cfg.residual.base_model
        key = (
            str(self.device),
            json.dumps(OmegaConf.to_container(model_cfg, resolve=True), sort_keys=True),
        )
        base = _BASE_MODELS.get(key)
        if base is None:
            base = get_model(model_cfg).to(self.device)
            base.requires_grad_(False)
            base.eval()
            _BASE_MODELS[key] = base
        self.base_policy = base

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Invalidate reset plans before the parent captures reset observations."""
        self.base_cache.invalidate(None if options is None else options.get("env_idx"))
        return super().reset(seed=seed, options=options)

    def _wrap_obs(
        self, raw_obs: dict[str, Any], infos: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        obs = super()._wrap_obs(raw_obs, infos)
        self._load_base()
        self.base_cache.refresh(obs, self.base_policy)
        result = dict(obs)
        result.update(self.base_cache.observation())
        if not self.cfg.residual.use_state:
            result.pop("states", None)
        return result

    def step(
        self, actions: torch.Tensor | None = None, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Execute one composed action and expose the next base condition."""
        executed = self.base_cache.compose(
            actions,
            self.residual_scale,
            gripper_mode=self.cfg.residual.get("gripper_mode", "residual"),
        )
        obs, reward, terminated, truncated, info = super().step(executed, auto_reset)
        info["residual_executed_action"] = executed.clone()
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions: torch.Tensor) -> tuple:
        """Require fresh observations between residual corrections."""
        if chunk_actions.shape[1] != 1:
            raise ValueError("Residual ManiSkill requires num_action_chunks=1.")
        return super().chunk_step(chunk_actions)
