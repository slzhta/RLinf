"""LeRobot/OpenPI data configuration for the Panda UMI PnP dataset."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import pnp_policy


@dataclasses.dataclass(frozen=True)
class LeRobotPnPDataConfig(DataConfigFactory):
    default_prompt: str | None = (
        "pick up the cube and place it at the target position"
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[pnp_policy.PnPInputs(model_type=model_config.model_type)],
            outputs=[pnp_policy.PnPOutputs()],
        )

        # Collector actions are already EE delta pose + absolute gripper.
        # Deliberately do not apply DeltaActions a second time.
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
