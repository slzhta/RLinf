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

from rlinf.workers.actor.fsdp_dagger_policy_worker import _apply_sft_drq


def test_drq_augments_offline_sft_forward_inputs():
    main_images = torch.arange(2 * 8 * 8 * 3).reshape(2, 8, 8, 3)
    extra_images = main_images.unsqueeze(1).clone()
    batch = {
        "forward_inputs": {
            "main_images": main_images.clone(),
            "extra_view_images": extra_images.clone(),
            "action": torch.zeros(2, 7),
        }
    }

    _apply_sft_drq(batch)

    augmented = batch["forward_inputs"]
    assert augmented["main_images"].shape == main_images.shape
    assert augmented["extra_view_images"].shape == extra_images.shape
    torch.testing.assert_close(augmented["action"], torch.zeros(2, 7))
