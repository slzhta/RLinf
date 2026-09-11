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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNConfig, CNNPolicy


@dataclass
class ResidualConfig(CNNConfig):
    """CNN configuration with frozen-base conditioning and active action channels."""

    enabled: bool = True
    base_horizon: int = 10
    residual_action_indices: list[int] = field(default_factory=lambda: list(range(6)))


class ResidualPolicy(CNNPolicy):
    """Reuse CNNPolicy's encoder, Gaussian PPO policy and value head.

    The native numeric observation channel holds the remaining base plan and
    its mask, followed by optional proprioception. PPO stores this condition
    together with the raw Gaussian action using the normal CNN rollout schema.
    """

    def __init__(self, cfg: ResidualConfig) -> None:
        indices = cfg.residual_action_indices
        if cfg.num_action_chunks != 1 or cfg.add_q_head or not cfg.add_value_head:
            raise ValueError("ResidualPolicy requires one-step PPO with a value head.")
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(i < 0 or i >= cfg.action_dim for i in indices)
        ):
            raise ValueError("Invalid residual_action_indices.")
        if cfg.binary_action_indices or cfg.action_scale is not None:
            raise ValueError(
                "Residual PPO requires the native continuous Gaussian policy."
            )
        cnn_cfg = deepcopy(cfg)
        cnn_cfg.action_dim = len(indices)
        for key in ("initial_logstd", "action_std_scale"):
            value = getattr(cnn_cfg, key)
            if isinstance(value, list) and value:
                if len(value) != cfg.action_dim:
                    raise ValueError(f"{key} must contain action_dim entries.")
                setattr(cnn_cfg, key, [value[i] for i in indices])
        condition_dim = cfg.base_horizon * (cfg.action_dim + 1)
        cnn_cfg.use_state = True
        cnn_cfg.state_dim = condition_dim + (cfg.state_dim if cfg.use_state else 0)
        super().__init__(cnn_cfg)
        self.residual_cfg = deepcopy(cfg)
        self.register_buffer("action_indices", torch.tensor(indices, dtype=torch.long))
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def _prepare_cnn_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        base = obs["base_actions"]
        mask = obs["base_action_mask"].to(base)
        if base.shape[1:] != (
            self.residual_cfg.base_horizon,
            self.residual_cfg.action_dim,
        ):
            raise ValueError("base_actions must match base_horizon and action_dim.")
        if mask.shape != base.shape[:2]:
            raise ValueError("base_action_mask must match base_actions.")
        parts = [(base * mask.unsqueeze(-1)).flatten(1), mask]
        if self.residual_cfg.use_state:
            states = obs.get("states")
            if states is None or states.shape != (
                base.shape[0],
                self.residual_cfg.state_dim,
            ):
                raise ValueError(
                    "use_state=True requires states with shape [B, state_dim]."
                )
            parts.append(states.to(base))
        extra = obs.get("extra_view_images")
        if self.cfg.image_num > 1 and (
            extra is None or extra.shape[1] != self.cfg.image_num - 1
        ):
            raise ValueError("Missing configured residual camera views.")
        return {**obs, "states": torch.cat(parts, dim=-1)}

    def _expand_actions(self, actions: torch.Tensor) -> torch.Tensor:
        shape = (*actions.shape[:-1], self.residual_cfg.action_dim)
        return actions.new_zeros(shape).index_copy(-1, self.action_indices, actions)

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Use the standard CNN PPO forward dispatch."""
        if forward_type != ForwardType.DEFAULT:
            raise NotImplementedError(
                "ResidualPolicy supports PPO default forward only."
            )
        return super().forward(forward_type=forward_type, **kwargs)

    def default_forward(
        self, forward_inputs: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Evaluate the native Gaussian on the saved active action channels."""
        inputs = dict(forward_inputs)
        inputs["action"] = inputs["action"].index_select(-1, self.action_indices)
        result = super().default_forward(forward_inputs=inputs, **kwargs)
        for key in ("logprobs", "entropy"):
            if key in result:
                result[key] = self._expand_actions(result[key])
        if "entropy" in result:
            # Preserve the single action-step axis for PPO's [B, 1] loss mask.
            # A [B] chunk entropy would broadcast with that mask to [B, B].
            result["entropy"] = result["entropy"].reshape(
                -1, 1, self.residual_cfg.action_dim
            )
        return result

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict[str, torch.Tensor], **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Add base conditions and delegate sampling/storage to CNNPolicy."""
        if not self.residual_cfg.enabled and kwargs.get("mode", "train") != "eval":
            raise ValueError("Disable residual only for evaluation.")
        actions, result = super().predict_action_batch(
            env_obs=self._prepare_cnn_obs(env_obs), **kwargs
        )
        actions = self._expand_actions(actions)
        result["prev_logprobs"] = self._expand_actions(result["prev_logprobs"])
        inputs = result["forward_inputs"]
        inputs["action"] = self._expand_actions(inputs["action"])
        result["forward_inputs"] = {
            key: value.detach().clone() for key, value in inputs.items()
        }
        if not self.residual_cfg.enabled:
            actions = torch.zeros_like(actions)
        return actions, result
