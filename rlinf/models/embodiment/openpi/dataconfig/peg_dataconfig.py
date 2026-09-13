# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""OpenPI single-wrist, relative-state Peg SFT configuration."""

import dataclasses
import pathlib

from openpi import transforms
from openpi.models.model import BaseModelConfig
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies.peg_policy import PegInputs, PegOutputs


@dataclasses.dataclass(frozen=True)
class LeRobotPegDataConfig(DataConfigFactory):
    default_prompt: str = (
        "Insert the green U-shaped peg into the matching hole in the blue board"
    )

    def create(
        self, assets_dirs: pathlib.Path, model_config: BaseModelConfig
    ) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=transforms.Group(
                inputs=[PegInputs()], outputs=[PegOutputs()]
            ),
            model_transforms=ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            ),
            use_quantile_norm=True,
        )
