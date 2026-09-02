# Copyright 2025 The RLinf Authors.
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

from gymnasium.envs.registration import register

from rlinf.envs.realworld.franka.franka_env import FrankaEnv as FrankaEnv
from rlinf.envs.realworld.franka.tasks.bottle import BottleEnv as BottleEnv
from rlinf.envs.realworld.franka.tasks.co_training_base_env import (
    FrankaCoTrainingBaseEnv as FrankaCoTrainingBaseEnv,
)
from rlinf.envs.realworld.franka.tasks.franka_bin_relocation import (
    FrankaBinRelocationEnv as FrankaBinRelocationEnv,
)
from rlinf.envs.realworld.franka.tasks.peg_insertion_env import (
    PegInsertionEnv as PegInsertionEnv,
)
from rlinf.envs.realworld.franka.tasks.pick_and_place_env import (
    FrankaPickAndPlaceEnv as FrankaPickAndPlaceEnv,
)
from rlinf.envs.realworld.franka.tasks.push_button_env import (
    FrankaPushButtonEnv as FrankaPushButtonEnv,
)

register(
    id="FrankaEnv-v1",
    entry_point="rlinf.envs.realworld.franka.franka_env:FrankaEnv",
)

register(
    id="PegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:PegInsertionEnv",
)

register(
    id="FrankaBinRelocationEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:FrankaBinRelocationEnv",
)

register(id="BottleEnv-v1", entry_point="rlinf.envs.realworld.franka.tasks:BottleEnv")

register(
    id="FrankaCoTrainingBaseEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:FrankaCoTrainingBaseEnv",
)

register(
    id="FrankaPushButtonEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:FrankaPushButtonEnv",
)

register(
    id="FrankaPickAndPlaceEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:FrankaPickAndPlaceEnv",
)
