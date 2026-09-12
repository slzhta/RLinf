# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Six tanh-Gaussian arm residuals and an independently BC-initialized gripper."""

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributions import Bernoulli, Normal

from rlinf.models.embodiment.residual_policy.gripper_cnn import (
    GripperCNN,
    GripperCNNConfig,
)
from rlinf.models.embodiment.residual_policy.residual_policy import (
    ResidualConfig,
    ResidualPolicy,
)


@dataclass
class SplitGripperConfig(ResidualConfig):
    """Optional split policy; original residual configurations are unchanged."""

    gripper_mode: str = "cnn"
    gripper_architecture: str = "independent"
    gripper_checkpoint: str | None = None
    gripper: dict[str, Any] = field(default_factory=dict)
    gripper_initial_logstd: float = -1.0
    gripper_logstd_range: list[float] = field(default_factory=lambda: [-4.0, 0.0])


class SplitGripperPolicy(ResidualPolicy):
    """Joint PPO with tanh-Gaussian arm and binary or scalar gripper control."""

    def __init__(self, cfg: SplitGripperConfig) -> None:
        if cfg.action_dim != 7 or list(cfg.residual_action_indices) != list(range(6)):
            raise ValueError(
                "Split gripper requires exactly six arm residuals and action_dim=7."
            )
        temperature = float(cfg.binary_action_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("binary_action_temperature must be finite and positive.")
        gripper_cfg = GripperCNNConfig(**cfg.gripper)
        if gripper_cfg.image_num > cfg.image_num:
            raise ValueError(
                "Gripper cannot use more cameras than the residual rollout stores."
            )
        if gripper_cfg.state_dim and (
            not cfg.use_state or gripper_cfg.state_dim != cfg.state_dim
        ):
            raise ValueError("Gripper state input must match residual proprioception.")
        super().__init__(cfg)
        self.gripper = GripperCNN(gripper_cfg)
        if gripper_cfg.output_mode == "scalar":
            bounds = cfg.gripper_logstd_range
            initial = float(cfg.gripper_initial_logstd)
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) for value in (*bounds, initial))
                or not bounds[0] <= initial <= bounds[1]
            ):
                raise ValueError(
                    "gripper_initial_logstd must lie in gripper_logstd_range."
                )
            self.gripper_logstd = torch.nn.Parameter(torch.tensor([[initial]]))
        if cfg.gripper_checkpoint:
            self.gripper.load_bc(cfg.gripper_checkpoint)

    def _prepare_cnn_obs(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Keep the base model's native seven-dimensional tensor contract, but
        # remove its gripper plan even from the residual/value conditioning.
        base = obs["base_actions"].clone()
        base[..., 6] = 0
        return super()._prepare_cnn_obs({**obs, "base_actions": base})

    def _gripper_obs(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        obs = {"main_images": inputs["main_images"]}
        if self.gripper.cfg.image_num > 1:
            obs["extra_view_images"] = inputs["extra_view_images"][
                :, : self.gripper.cfg.image_num - 1
            ]
        if self.gripper.cfg.state_dim:
            # ResidualPolicy stores [base plan, mask, original proprioception].
            obs["states"] = inputs["states"][:, -self.gripper.cfg.state_dim :]
        return obs

    def _gripper_distribution(self, logits: torch.Tensor) -> Bernoulli | Normal:
        """Build the binary distribution or the scalar's latent Gaussian."""
        if self.gripper.cfg.output_mode == "scalar":
            low, high = self.residual_cfg.gripper_logstd_range
            std = self.gripper_logstd.clamp(low, high).exp().expand_as(logits)
            return Normal(logits, std)
        return Bernoulli(
            logits=logits / float(self.residual_cfg.binary_action_temperature)
        )

    def _gripper_statistics(
        self, logits: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._gripper_distribution(logits)
        if self.gripper.cfg.output_mode == "scalar":
            return self._continuous_action_statistics(
                logits, distribution.stddev, action
            )
        return (
            distribution.log_prob((action > 0).to(logits.dtype)),
            distribution.entropy(),
        )

    def default_forward(
        self, forward_inputs: dict[str, torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Recompute both distributions on the exact actions saved at rollout."""
        result = super().default_forward(forward_inputs=forward_inputs, **kwargs)
        if "logprobs" in result or "entropy" in result:
            logprobs, gripper_entropy = self._gripper_statistics(
                self.gripper(self._gripper_obs(forward_inputs)),
                forward_inputs["action"][..., 6:7],
            )
            if "logprobs" in result:
                result["logprobs"] = torch.cat(
                    (result["logprobs"][..., :6], logprobs),
                    -1,
                )
            if "entropy" in result:
                entropy = result["entropy"].reshape(-1, 1, 7)
                result["entropy"] = torch.cat(
                    (entropy[..., :6], gripper_entropy.unsqueeze(1)), -1
                )
        return result

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: dict[str, torch.Tensor], **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Save the sampled gripper command before environment thresholding."""
        if kwargs.get("return_obs", True) is False:
            raise ValueError("Split gripper requires return_obs=True for PPO replay.")
        actions, result = super().predict_action_batch(env_obs, **kwargs)
        inputs = result["forward_inputs"]
        # CNNPolicy stores replay observations on CPU even for CUDA rollout.
        device = next(self.gripper.parameters()).device
        gripper_inputs = {
            key: value.to(device) for key, value in self._gripper_obs(inputs).items()
        }
        logits = self.gripper(gripper_inputs)
        distribution = self._gripper_distribution(logits)
        if self.gripper.cfg.output_mode == "scalar":
            latent = (
                distribution.sample()
                if kwargs.get("mode", "train") == "train"
                else logits
            )
            command = torch.tanh(latent)
        else:
            binary = (
                distribution.sample()
                if kwargs.get("mode", "train") == "train"
                else (logits >= 0).to(logits)
            )
            command = binary * 2 - 1
        logprobs, _ = self._gripper_statistics(logits, command)
        actions = torch.cat((actions[..., :6], command.to(actions).unsqueeze(1)), -1)
        inputs["action"] = torch.cat(
            (inputs["action"][..., :6], command.to(inputs["action"])), -1
        )
        result["prev_logprobs"] = torch.cat(
            (
                result["prev_logprobs"][..., :6],
                logprobs.to(result["prev_logprobs"]),
            ),
            -1,
        )
        return actions, result
