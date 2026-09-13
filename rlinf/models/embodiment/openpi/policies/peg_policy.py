# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Single-wrist six-axis transforms matching Peg OpenPI SFT preprocessing."""

from dataclasses import dataclass

import numpy as np
from openpi import transforms


@dataclass(frozen=True)
class PegInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (6,):
            raise ValueError(
                f"Peg requires six-dimensional target-relative state, got {state.shape}"
            )
        image = np.asarray(data["observation/image"])
        if image.ndim != 3:
            raise ValueError(f"Peg requires one RGB image, got {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.shape[-1] != 3:
            raise ValueError("Peg image must be RGB.")
        result = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 6:
                raise ValueError("Peg actions must have six dimensions.")
            result["actions"] = transforms.pad_to_dim(actions, 32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class PegOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :6]}
