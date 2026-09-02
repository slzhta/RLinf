"""OpenPI transforms for Panda UMI pick-and-place demonstrations."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image).squeeze()
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"expected a 3-D image, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"expected RGB image, got {image.shape}")
    return image.astype(np.uint8, copy=False)


@dataclasses.dataclass(frozen=True)
class PnPInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    state_dim: int = 14
    env_action_dim: int = 7
    model_action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected state dim {self.state_dim}, got {state.shape}")

        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if base_image.shape != wrist_image.shape:
            raise ValueError(
                f"base/wrist image shapes differ: {base_image.shape}, {wrist_image.shape}"
            )

        inputs = {
            "state": transforms.pad_to_dim(state, self.model_action_dim),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != self.env_action_dim:
                raise ValueError(
                    f"expected action dim {self.env_action_dim}, got {actions.shape}"
                )
            inputs["actions"] = transforms.pad_to_dim(
                actions, self.model_action_dim
            )
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class PnPOutputs(transforms.DataTransformFn):
    env_action_dim: int = 7

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.env_action_dim])}
