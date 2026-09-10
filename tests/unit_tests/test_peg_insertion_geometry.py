"""CPU checks for target-frame peg reset randomization."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from rlinf.envs.peg_insertion_geometry import PegInsertionGeometry


class PegResetTests(unittest.TestCase):
    def test_default_randomization(self):
        geometry = PegInsertionGeometry()
        self.assertAlmostEqual(geometry.random_xy, 0.05)
        self.assertAlmostEqual(np.rad2deg(geometry.random_yaw), 10.0)
        self.assertGreaterEqual(geometry.workspace_xy, geometry.random_xy)

    def test_sampled_reset_bounds_and_coverage(self):
        geometry = PegInsertionGeometry(
            target_ee_pose=[0.6278, 0.0982, -0.0022, 3.0966, 0.0119, -0.0119],
            workspace_xy=0.15,
            rotation_limits=[0.01, 0.01, 0.5],
        )
        rng = np.random.default_rng(1234)
        poses = np.stack([geometry.reset_pose(rng) for _ in range(1024)])
        offsets = (poses[:, :3, 3] - geometry.target[:3, 3]) @ geometry.insertion_rotation
        rotations = Rotation.from_matrix(
            poses[:, :3, :3] @ geometry.target[:3, :3].T
        ).as_rotvec()
        angles = rotations @ geometry.insertion_rotation[:, 2]
        self.assertTrue(np.all(np.abs(offsets[:, :2]) <= 0.05))
        np.testing.assert_allclose(offsets[:, 2], 0.10, atol=1e-12)
        self.assertTrue(np.all(np.abs(angles) <= np.deg2rad(10.0)))
        np.testing.assert_allclose(
            rotations, angles[:, None] * geometry.insertion_rotation[:, 2], atol=1e-12
        )
        self.assertTrue(np.all(offsets[:, :2].min(axis=0) < -0.045))
        self.assertTrue(np.all(offsets[:, :2].max(axis=0) > 0.045))
        self.assertLess(angles.min(), np.deg2rad(-9.0))
        self.assertGreater(angles.max(), np.deg2rad(9.0))
        np.testing.assert_allclose(geometry.clip_target(poses), poses, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
