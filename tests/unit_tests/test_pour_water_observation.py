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

"""Pouring uses the PnP camera/state contract without a gripper action."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml


def test_pour_dualview_absolute_state():
    pytest.importorskip("sapien")
    from rlinf.envs.maniskill.tasks.digital_twin.pour_water import (
        PourWaterDigitalTwinEnv,
    )

    env = object.__new__(PourWaterDigitalTwinEnv)
    env.num_envs = 1
    pose = torch.eye(4).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([0.49, 0.12, 0.18])
    joints = torch.tensor([[0.1, 0.2, 0.3, -1.0, 0.5, 1.2, 0.7, 0.02, 0.02]])
    env.agent = SimpleNamespace(
        robot=SimpleNamespace(get_qpos=lambda: joints),
        ee_pose_at_robot_base=SimpleNamespace(to_transformation_matrix=lambda: pose),
    )
    cameras = {
        "3rdview_camera": {"rgb": torch.full((1, 48, 64, 3), 17, dtype=torch.uint8)},
        "hand_camera": {"rgb": torch.full((1, 48, 64, 3), 231, dtype=torch.uint8)},
    }
    observation = env._build_extracted_obs({"sensor_data": cameras})
    assert observation["main_images"].shape == (1, 224, 224, 3)
    assert observation["extra_view_images"].shape == (1, 1, 224, 224, 3)
    assert observation["main_images"][0, 112, 112, 0] == 17
    assert observation["extra_view_images"][0, 0, 112, 112, 0] == 231
    expected = torch.cat(
        [joints[:, :7], pose[:, :3, 3], torch.zeros(1, 3), -torch.ones(1, 1)], dim=1
    )
    torch.testing.assert_close(observation["states"], expected)
    with pytest.raises(ValueError, match="Wrist RGB"):
        env._build_extracted_obs(
            {"sensor_data": {"3rdview_camera": cameras["3rdview_camera"]}}
        )


def test_pour_policy_and_real_state_config():
    root = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    config = yaml.safe_load(
        (root / "co_rl_pour_water_async_ppo_cnn_wrist_state.yaml").read_text()
    )
    real = yaml.safe_load((root / "env/realworld_pour_water_co_rl.yaml").read_text())
    assert config["actor"]["model"]["image_num"] == 2
    assert config["actor"]["model"]["state_dim"] == 14
    assert config["actor"]["model"]["action_dim"] == 6
    assert config["actor"]["model"]["binary_action_indices"] == []
    assert real["main_image_key"] == "wrist_2"
    assert real["state_keys"] == [
        "arm_joint_position",
        "tcp_pose",
        "gripper_open_state",
    ]
