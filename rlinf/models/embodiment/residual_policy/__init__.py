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

import torch
from omegaconf import DictConfig, OmegaConf


def get_model(
    cfg: DictConfig, torch_dtype: torch.dtype = torch.float32
) -> torch.nn.Module:
    """Build the small PPO policy; the frozen base belongs to the environment."""
    from rlinf.models.embodiment.residual_policy.residual_policy import (
        ResidualConfig,
        ResidualPolicy,
    )

    model_config = ResidualConfig()
    model_config.update_from_dict(OmegaConf.to_container(cfg, resolve=True))
    return ResidualPolicy(model_config)
