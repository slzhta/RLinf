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

"""The PnP source/target tray geometry shared by tabletop tasks."""

import os
from pathlib import Path

import numpy as np
import sapien
from sapien.physx import PhysxMaterial


class RectangularTrayMixin:
    TAPE_COLOR = [0.74117647, 0.54509804, 0.10980392, 1.0]
    PLATE_COLOR = [0.94901961, 0.94509804, 0.93333333, 1.0]
    PLATE_LENGTH = 0.24
    PLATE_WIDTH = 0.175
    PLATE_DEPTH = 0.015
    PLATE_BOTTOM_THICKNESS = 0.002
    PLATE_WALL_THICKNESS = 0.005
    PLATE_CORNER_RADIUS = 0.03
    PLATE_TOP_RIM_WIDTH = 0.005
    TAPE_WIDTH = 0.04
    TAPE_LENGTH = 0.125
    TAPE_THICKNESS = 0.0005
    DEFAULT_TRAY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @classmethod
    def _floor_top_local_z(cls) -> float:
        return -cls.PLATE_DEPTH / 2.0 + cls.PLATE_BOTTOM_THICKNESS

    def _build_tray(
        self,
        name: str,
        has_tape: bool,
        initial_position: np.ndarray,
    ):
        builder = self.scene.create_actor_builder()
        collision_material = PhysxMaterial(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        bottom_z = -self.PLATE_DEPTH / 2.0
        builder.add_box_collision(
            pose=sapien.Pose(
                p=[0.0, 0.0, bottom_z + self.PLATE_BOTTOM_THICKNESS / 2.0]
            ),
            half_size=[
                self.PLATE_LENGTH / 2.0,
                self.PLATE_WIDTH / 2.0,
                self.PLATE_BOTTOM_THICKNESS / 2.0,
            ],
            material=collision_material,
        )
        self._add_tray_wall_collisions(builder, collision_material)
        builder.add_visual_from_file(
            filename=str(self._get_tray_mesh_path()),
            material=sapien.render.RenderMaterial(base_color=self.PLATE_COLOR),
        )
        if has_tape:
            self._add_tape_strips(builder)
        builder.initial_pose = sapien.Pose(
            p=np.asarray(initial_position, dtype=np.float32).tolist(),
            q=self.DEFAULT_TRAY_QUAT.tolist(),
        )
        return builder.build_kinematic(name=name)

    def _add_tray_wall_collisions(
        self,
        builder,
        collision_material: PhysxMaterial,
    ):
        wall_half_z = self.PLATE_DEPTH / 2.0
        long_wall_half_size = [
            self.PLATE_LENGTH / 2.0,
            self.PLATE_WALL_THICKNESS / 2.0,
            wall_half_z,
        ]
        short_wall_half_size = [
            self.PLATE_WALL_THICKNESS / 2.0,
            self.PLATE_WIDTH / 2.0 - self.PLATE_WALL_THICKNESS,
            wall_half_z,
        ]
        for y_sign in (-1.0, 1.0):
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[
                        0.0,
                        y_sign
                        * (self.PLATE_WIDTH / 2.0 - self.PLATE_WALL_THICKNESS / 2.0),
                        0.0,
                    ]
                ),
                half_size=long_wall_half_size,
                material=collision_material,
            )
        for x_sign in (-1.0, 1.0):
            builder.add_box_collision(
                pose=sapien.Pose(
                    p=[
                        x_sign
                        * (self.PLATE_LENGTH / 2.0 - self.PLATE_WALL_THICKNESS / 2.0),
                        0.0,
                        0.0,
                    ]
                ),
                half_size=short_wall_half_size,
                material=collision_material,
            )

    def _add_tape_strips(self, builder):
        tape_material = sapien.render.RenderMaterial(base_color=self.TAPE_COLOR)
        tape_collision_material = PhysxMaterial(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        tape_center_x = (
            self.PLATE_LENGTH / 2.0 - self.PLATE_WALL_THICKNESS - self.TAPE_WIDTH / 2.0
        )
        tape_z = self._floor_top_local_z() + self.TAPE_THICKNESS / 2.0
        tape_half_size = [
            self.TAPE_WIDTH / 2.0,
            self.TAPE_LENGTH / 2.0,
            self.TAPE_THICKNESS / 2.0,
        ]
        for x_sign in (-1.0, 1.0):
            tape_pose = sapien.Pose(p=[x_sign * tape_center_x, 0.0, tape_z])
            builder.add_box_collision(
                pose=tape_pose,
                half_size=tape_half_size,
                material=tape_collision_material,
            )
            builder.add_box_visual(
                pose=tape_pose,
                half_size=tape_half_size,
                material=tape_material,
            )

    @classmethod
    def _get_tray_mesh_path(cls) -> Path:
        mesh_dir = Path("/tmp/rlinf_digital_twin_assets")
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / "rectangular_tray_24x17p5x1p5.obj"
        if mesh_path.is_file():
            return mesh_path

        tmp_path = mesh_path.with_suffix(".obj.tmp")
        tmp_path.write_text(cls._build_tray_obj(), encoding="utf-8")
        os.replace(tmp_path, mesh_path)
        return mesh_path

    @classmethod
    def _build_tray_obj(cls) -> str:
        outer = cls._rounded_rect_points(
            half_x=cls.PLATE_LENGTH / 2.0,
            half_y=cls.PLATE_WIDTH / 2.0,
            radius=cls.PLATE_CORNER_RADIUS,
        )
        inner = cls._rounded_rect_points(
            half_x=cls.PLATE_LENGTH / 2.0 - cls.PLATE_TOP_RIM_WIDTH,
            half_y=cls.PLATE_WIDTH / 2.0 - cls.PLATE_TOP_RIM_WIDTH,
            radius=cls.PLATE_CORNER_RADIUS - cls.PLATE_TOP_RIM_WIDTH,
        )
        bottom_z = -cls.PLATE_DEPTH / 2.0
        top_z = cls.PLATE_DEPTH / 2.0
        floor_z = bottom_z + cls.PLATE_BOTTOM_THICKNESS

        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        outer_bottom = cls._add_ring(vertices, outer, bottom_z)
        outer_top = cls._add_ring(vertices, outer, top_z)
        inner_top = cls._add_ring(vertices, inner, top_z)
        inner_floor = cls._add_ring(vertices, inner, floor_z)

        cls._add_side_faces(faces, outer_bottom, outer_top)
        cls._add_side_faces(faces, inner_top, inner_floor)
        cls._add_ring_faces(faces, outer_top, inner_top)
        cls._add_cap_faces(faces, vertices, outer_bottom, bottom_z, upward=False)
        cls._add_cap_faces(faces, vertices, inner_floor, floor_z, upward=True)

        lines = ["# RLinf digital-twin rounded tray mesh"]
        lines.extend(f"v {x:.8f} {y:.8f} {z:.8f}" for x, y, z in vertices)
        lines.extend("f " + " ".join(str(idx + 1) for idx in face) for face in faces)
        return "\n".join(lines) + "\n"

    @classmethod
    def _rounded_rect_points(
        cls,
        half_x: float,
        half_y: float,
        radius: float,
        segments_per_corner: int = 12,
    ) -> list[tuple[float, float]]:
        radius = min(radius, half_x, half_y)
        centers = [
            (half_x - radius, half_y - radius),
            (-half_x + radius, half_y - radius),
            (-half_x + radius, -half_y + radius),
            (half_x - radius, -half_y + radius),
        ]
        angle_ranges = [
            (0.0, np.pi / 2.0),
            (np.pi / 2.0, np.pi),
            (np.pi, 3.0 * np.pi / 2.0),
            (3.0 * np.pi / 2.0, 2.0 * np.pi),
        ]

        points = []
        for center, angle_range in zip(centers, angle_ranges):
            angles = np.linspace(
                angle_range[0],
                angle_range[1],
                segments_per_corner,
                endpoint=False,
            )
            points.extend(
                (
                    center[0] + radius * float(np.cos(angle)),
                    center[1] + radius * float(np.sin(angle)),
                )
                for angle in angles
            )
        return points

    @staticmethod
    def _add_ring(
        vertices: list[list[float]],
        points: list[tuple[float, float]],
        z: float,
    ) -> list[int]:
        indices = []
        for x, y in points:
            indices.append(len(vertices))
            vertices.append([x, y, z])
        return indices

    @staticmethod
    def _add_side_faces(faces: list[list[int]], lower: list[int], upper: list[int]):
        n = len(lower)
        for i in range(n):
            j = (i + 1) % n
            faces.append([lower[i], lower[j], upper[j], upper[i]])

    @staticmethod
    def _add_ring_faces(faces: list[list[int]], outer: list[int], inner: list[int]):
        n = len(outer)
        for i in range(n):
            j = (i + 1) % n
            faces.append([outer[i], outer[j], inner[j], inner[i]])

    @staticmethod
    def _add_cap_faces(
        faces: list[list[int]],
        vertices: list[list[float]],
        ring: list[int],
        z: float,
        upward: bool,
    ):
        center = len(vertices)
        xy = np.mean(np.asarray([vertices[idx][:2] for idx in ring]), axis=0)
        vertices.append([float(xy[0]), float(xy[1]), z])
        n = len(ring)
        for i in range(n):
            j = (i + 1) % n
            face = [center, ring[i], ring[j]] if upward else [center, ring[j], ring[i]]
            faces.append(face)
