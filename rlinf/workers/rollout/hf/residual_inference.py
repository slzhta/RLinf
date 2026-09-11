# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen OpenPI inference for paired real/sim residual rollouts."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.workers.rollout.hf.base_action_cache import BaseActionCache


class RolloutResidualInference:
    """Own base weights and caches separately from the synchronized PPO policy.

    One instance belongs to one rollout worker. The actor never constructs it.
    The existing PPO forward_inputs retain unscaled sampled residuals and exact base
    conditions; only the separately returned actions are composed for execution.
    """

    def __init__(self, cfg: DictConfig, device: Any, base_policy: Any = None):
        self.cfg = cfg
        self.device = device
        if base_policy is None:
            from rlinf.models.embodiment.openpi import get_model

            # Resolve '~' on the rollout host, not the driver. The two GPU
            # hosts use different Linux users but the same home-relative path.
            base_cfg = OmegaConf.create(
                OmegaConf.to_container(cfg.base_model, resolve=True)
            )
            base_cfg.model_path = str(Path(base_cfg.model_path).expanduser())
            base_policy = get_model(base_cfg).to(device)
            base_policy.requires_grad_(False)
            base_policy.eval()
        self.base_policy = base_policy
        self.caches: dict[str, BaseActionCache] = {}
        self.scale = torch.tensor(
            list(cfg.actor.model.residual_action_scale),
            dtype=torch.float32,
            device=device,
        )

    @torch.no_grad()
    def predict(
        self,
        policy: Any,
        env_obs: dict[str, Any],
        *,
        mode: str = "train",
        consume: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Infer a conditioned residual and optionally consume one base action."""
        obs = dict(env_obs)
        reset_mask = obs.pop("_residual_reset_mask", None)
        if reset_mask is None:
            raise ValueError(
                "Residual rollout requires an explicit environment reset mask."
            )
        batch_size = obs["main_images"].shape[0]
        mask = torch.as_tensor(reset_mask, device=self.device, dtype=torch.bool)
        if mask.shape != (batch_size,):
            raise ValueError("Residual reset mask must have shape [num_envs].")
        if mode not in self.caches:
            self.caches[mode] = BaseActionCache(
                batch_size,
                self.cfg.actor.model.base_horizon,
                self.cfg.actor.model.action_dim,
                self.device,
            )
        cache = self.caches[mode]
        if len(cache.position) != batch_size:
            raise ValueError("Residual rollout batch size changed within a domain.")
        cache.invalidate(mask.nonzero(as_tuple=True)[0])
        cache.refresh(obs, self.base_policy)
        conditioned = {**obs, **cache.observation()}
        actions, result = policy.predict_action_batch(
            env_obs=conditioned, mode=mode, **kwargs
        )
        if not consume:
            # The last PPO request only records V(s). The same observation is
            # sent at the start of the next epoch; its base action is unconsumed.
            return actions, result
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (batch_size, 1, 7):
            raise ValueError("Residual rollout requires one seven-dimensional action.")
        executed = cache.compose(actions[:, 0], self.scale, gripper_mode="cnn")
        return executed.unsqueeze(1), result
