# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check that global-batch BC gradients and portable weights survive two GPUs."""

from copy import deepcopy

import pytest
import torch

from rlinf.models.embodiment.residual_policy.gripper_cnn import (
    GripperCNN,
    GripperCNNConfig,
)
from toolkits.residual.train_gripper_bc import training_replica


def test_parallel_requires_cuda():
    model = GripperCNN(GripperCNNConfig())
    assert training_replica(model, torch.device("cpu"), False) is model
    with pytest.raises(ValueError, match="cuda:0"):
        training_replica(model, torch.device("cpu"), True)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="Requires two free visible CUDA GPUs."
)
def test_two_gpu_global_batch_gradients_and_bc_export(tmp_path):
    torch.manual_seed(1234)
    cfg = GripperCNNConfig(image_num=2, image_size=32, hidden_dim=32)
    single = GripperCNN(cfg).cuda(0)
    model = deepcopy(single)
    parallel = training_replica(model, torch.device("cuda:0"), True)
    obs = {
        "main_images": torch.randint(
            256, (8, 32, 32, 3), dtype=torch.uint8, device="cuda:0"
        ),
        "extra_view_images": torch.randint(
            256, (8, 1, 32, 32, 3), dtype=torch.uint8, device="cuda:0"
        ),
        "states": torch.randn(8, 14, device="cuda:0"),
    }
    target = torch.tensor([[0.0], [1.0]] * 4, device="cuda:0")
    # Disable TF32 for the strict single/full-batch reference comparison.
    with torch.backends.cudnn.flags(allow_tf32=False, deterministic=True):
        left, right = single(obs), parallel(obs)
        torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
        torch.nn.functional.binary_cross_entropy_with_logits(left, target).backward()
        torch.nn.functional.binary_cross_entropy_with_logits(right, target).backward()
    for a, b in zip(single.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=1e-3)
    model.save_bc(tmp_path / "bc.pt")
    loaded = GripperCNN(cfg)
    loaded.load_bc(tmp_path / "bc.pt")
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name].cpu())
