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

"""CPU checks for pouring action/state conventions and full-sphere containment."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.pour_water_geometry import PourWaterGeometry, ball_inside_cup


class PourWaterTests(unittest.TestCase):
    def test_containment_requires_whole_sphere(self):
        positions = [
            [0, 0, 0.009],
            [0, 0, 0.04],
            [0, 0, 0.088],
            [0.025, 0, 0.02],
            [0, 0, 0],
            [0.09, 0, 0.04],
        ]
        np.testing.assert_array_equal(
            ball_inside_cup(positions), [True, True, False, False, False, False]
        )

    def test_state_relative_to_reference(self):
        geometry = PourWaterGeometry()
        np.testing.assert_allclose(
            geometry.relative_state(geometry.target), 0, atol=1e-7
        )
        shifted = geometry.target.copy()
        shifted[:3, 3] += geometry.target[:3, :3] @ [0.01, 0.02, 0.03]
        np.testing.assert_allclose(
            geometry.relative_state(shifted)[:3], [0.01, 0.02, 0.03]
        )

    def test_action_matches_pnp_euler_increment(self):
        geometry = PourWaterGeometry()
        action = np.array([0.2, -0.1, 0.1, 0.2, -0.3, 0.4])
        actual = geometry.action_target(geometry.target, action)
        np.testing.assert_allclose(
            actual[:3, 3], geometry.target[:3, 3] + 0.01 * action[:3]
        )
        np.testing.assert_allclose(
            actual[:3, :3],
            Rotation.from_euler("xyz", 0.05 * action[3:]).as_matrix()
            @ geometry.target[:3, :3],
            atol=1e-12,
        )

    def test_reset_and_batch_clipping(self):
        geometry = PourWaterGeometry()
        rng = np.random.default_rng(12)
        poses = np.stack([geometry.reset_pose(rng) for _ in range(32)])
        np.testing.assert_allclose(geometry.clip_target(poses), poses, atol=1e-12)
        extreme = geometry.action_target(poses, np.ones((32, 6)) * 100)
        self.assertEqual(extreme.shape, (32, 4, 4))
        with self.assertRaises(ValueError):
            geometry.action_target(poses, np.zeros((32, 7)))
        with self.assertRaises(ValueError):
            geometry.action_target(poses, np.full((32, 6), np.nan))

    def test_tilt_allowed_without_unlimited_roll_yaw(self):
        geometry = PourWaterGeometry()
        pour = geometry.target.copy()
        pour[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0.60, 0]).as_matrix()
        np.testing.assert_allclose(geometry.clip_target(pour), pour, atol=1e-12)
        with self.assertRaises(ValueError):
            PourWaterGeometry(reset_position_random_range=[1, 1, 1])


if __name__ == "__main__":
    unittest.main()
