# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot embedded-image decoding must preserve BC frame/action alignment."""

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from rlinf.data.gripper_bc_dataset import GripperBCDataset

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def write_episode(path, indices=None, labels=None):
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(4)[:, None, None]
    records = []
    for image in images:
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="PNG")
        records.append({"bytes": stream.getvalue(), "path": "unused.png"})
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = [-1, -1, 1, 1] if labels is None else labels
    states = np.arange(56, dtype=np.float32).reshape(4, 14)
    table = pa.table(
        {
            "image": records,
            "wrist_image": records,
            "state": states.tolist(),
            "actions": actions.tolist(),
            "episode_index": [3] * 4 if indices is None else indices,
        }
    )
    pq.write_table(table, path)
    return images, states, actions


def test_parquet_and_npz_give_identical_bc_observations_and_targets(tmp_path):
    parquet = tmp_path / "episode_000003.parquet"
    images, states, actions = write_episode(parquet)
    npz = tmp_path / "episode_000003.npz"
    np.savez(
        npz, base_images=images, wrist_images=images, states=states, actions=actions
    )
    raw = GripperBCDataset([npz], ["base_images", "wrist_images"], 14)
    lerobot = GripperBCDataset(
        [parquet], ["image", "wrist_image"], 14, state_key="state"
    )
    assert len(lerobot) == 4
    for i in range(4):
        left, right = raw[i], lerobot[i]
        for key in left[0]:
            torch.testing.assert_close(left[0][key], right[0][key])
        torch.testing.assert_close(left[1], right[1])
        torch.testing.assert_close(left[2], right[2])
    vision_only = GripperBCDataset([parquet], ["image", "wrist_image"], 0)
    assert "states" not in vision_only[0][0]


def test_parquet_rejects_multiple_episodes_and_normalized_actions(tmp_path):
    path = tmp_path / "episode_000003.parquet"
    write_episode(path, indices=[0, 0, 1, 1])
    with pytest.raises(ValueError, match="one episode"):
        GripperBCDataset([path], ["image", "wrist_image"], 14, state_key="state")
    write_episode(path, labels=[-0.2, -0.2, 0.8, 0.8])
    with pytest.raises(ValueError, match="encoding"):
        GripperBCDataset([path], ["image", "wrist_image"], 14, state_key="state")
