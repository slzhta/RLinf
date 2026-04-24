# TODO(liangzhi): The import logit may be change, and the path should be change
import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


TABLE_SCALE = 1.0
TABLE_VISUAL_ROTATION = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
# The new GLB is already authored close to world scale. Its imported bounds are
# approximately x=1.2m, y=1.2m, z=0.5255m, with the bottom face at z=0.
TABLE_COLLISION_SIZE = np.array([1.2, 1.2, 0.525499], dtype=np.float32)
TABLE_POSE = sapien.Pose(
    p=[-0.12, 0, -float(TABLE_COLLISION_SIZE[2])],
    q=euler2quat(0, 0, np.pi / 2),
)


class TableSceneBuilder(SceneBuilder):
    """A simple scene builder that adds a table to the scene such that the height of the table is at 0, and
    gives reasonable initial poses for robots."""

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)).resolve().parent / "assets" / "tables"
        table_model_file = str(model_dir / "table.glb")
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, float(TABLE_COLLISION_SIZE[2] / 2)]),
            half_size=tuple((TABLE_COLLISION_SIZE / 2).tolist()),
        )
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[TABLE_SCALE] * 3,
            pose=TABLE_VISUAL_ROTATION,
        )
        builder.initial_pose = TABLE_POSE
        table = builder.build_kinematic(name="table-workspace")
        self.table_length, self.table_width, self.table_height = map(
            float, TABLE_COLLISION_SIZE
        )
        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-float(self.table_height),
        )
        self.table = table
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        self.table.set_pose(TABLE_POSE)
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_umi":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
            )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
            self.env.sync_gpu_articulation_state()
