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

from dataclasses import dataclass, field

import numpy as np

from .co_training_base_env import FrankaCoTrainingBaseConfig, FrankaCoTrainingBaseEnv


@dataclass
class FrankaPushButtonConfig(FrankaCoTrainingBaseConfig):
    task_description: str = "reach the button target and press it"
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.015, 0.015, 0.015, 0.2, 0.2, 0.2])
    )
    use_target_controller: bool = True
    enable_gripper_penalty: bool = False
    joint_reset_qpos: list[float] = field(
        default_factory=lambda: [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, np.pi / 4]
    )


class FrankaPushButtonEnv(FrankaCoTrainingBaseEnv):
    CONFIG_CLS = FrankaPushButtonConfig

    def reset(self, joint_reset=False, **kwargs):
        return super().reset(joint_reset=joint_reset, **kwargs)
