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

from typing import Any

import torch


class BaseActionCache:
    """Keep independent frozen-policy action chunks for vectorized environments."""

    def __init__(
        self, num_envs: int, horizon: int, action_dim: int, device: torch.device
    ) -> None:
        if num_envs < 1 or horizon < 1 or action_dim < 1:
            raise ValueError("Cache dimensions must be positive.")
        self.horizon = horizon
        self.actions = torch.zeros(num_envs, horizon, action_dim, device=device)
        self.position = torch.full(
            (num_envs,), horizon, device=device, dtype=torch.long
        )

    def invalidate(self, env_idx: torch.Tensor | None = None) -> None:
        """Clear only reset environments, preserving other plans and phases."""
        if env_idx is None:
            self.position.fill_(self.horizon)
        else:
            self.position[env_idx] = self.horizon

    @torch.no_grad()
    def refresh(self, obs: dict[str, Any], base_policy: Any) -> None:
        """Infer only exhausted plans using the base policy's unchanged eval path."""
        indices = (self.position >= self.horizon).nonzero(as_tuple=True)[0]
        if not indices.numel():
            return
        subset = {}
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                subset[key] = value.index_select(0, indices.to(value.device))
            elif isinstance(value, list):
                subset[key] = [value[i] for i in indices.tolist()]
            else:
                subset[key] = value
        subset.setdefault("wrist_images", None)
        subset.setdefault("extra_view_images", None)
        actions, _ = base_policy.predict_action_batch(
            env_obs=subset, mode="eval", compute_values=False
        )
        actions = torch.as_tensor(actions, device=self.actions.device).float()
        expected = (indices.numel(), self.horizon, self.actions.shape[-1])
        if (
            actions.ndim != 3
            or actions.shape[0] != expected[0]
            or actions.shape[1] < self.horizon
            or actions.shape[2] != expected[2]
        ):
            raise ValueError(
                f"Base actions must supply {expected}, got {actions.shape}."
            )
        actions = actions[:, : self.horizon]
        if not torch.isfinite(actions).all():
            raise ValueError("Frozen base produced nonfinite actions.")
        self.actions[indices] = actions
        self.position[indices] = 0

    def observation(self) -> dict[str, torch.Tensor]:
        """Return independent, zero-padded remaining plans for rollout and PPO updates."""
        offsets = torch.arange(self.horizon, device=self.actions.device)
        indices = self.position[:, None] + offsets
        mask = indices < self.horizon
        remaining = self.actions.gather(
            1, indices.clamp(max=self.horizon - 1).unsqueeze(-1).expand_as(self.actions)
        )
        return {
            "base_actions": remaining * mask.unsqueeze(-1),
            "base_action_mask": mask.clone(),
        }

    def compose(self, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Compose normalized controller deltas, then consume one cached action."""
        if (self.position >= self.horizon).any():
            raise RuntimeError("Refresh the base action cache before stepping.")
        residual = torch.as_tensor(residual, device=self.actions.device).float()
        if (
            residual.shape != self.actions[:, 0].shape
            or not torch.isfinite(residual).all()
        ):
            raise ValueError(
                "Residual actions must be finite with shape [num_envs, action_dim]."
            )
        current = self.actions[
            torch.arange(len(self.position), device=self.actions.device), self.position
        ]
        executed = (current + scale * residual.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)
        self.position += 1
        return executed
