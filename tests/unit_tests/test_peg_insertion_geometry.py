"""CPU checks for target-frame peg reset and success geometry."""

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


class PegSuccessTests(unittest.TestCase):
    def test_default_success_tolerances(self):
        geometry = PegInsertionGeometry()
        self.assertAlmostEqual(geometry.success_xy, 0.01)
        self.assertAlmostEqual(geometry.success_z, 0.01)
        self.assertAlmostEqual(np.rad2deg(geometry.success_angle), 5.0)
        self.assertEqual(geometry.success_hold_steps, 3)

    def test_success_requires_all_tolerances_in_insertion_frame(self):
        geometry = PegInsertionGeometry(
            target_ee_pose=[0.6278, 0.0982, -0.0022, 3.0966, 0.0119, -0.0119],
        )
        cases = [
            ([0.006, 0.007, 0.0099], 4.999, True),
            ([0.0, 0.0, -0.0099], 0.0, True),
            ([0.008, 0.008, 0.0], 0.0, False),
            ([0.0101, 0.0, 0.0], 0.0, False),
            ([0.0, 0.0, 0.0101], 0.0, False),
            ([0.0, 0.0, -0.0101], 0.0, False),
            ([0.0, 0.0, 0.0], 5.001, False),
        ]
        for offset, degrees, expected in cases:
            with self.subTest(offset=offset, degrees=degrees):
                current = geometry.target.copy()
                current[:3, 3] += geometry.insertion_rotation @ np.asarray(offset)
                rotvec = np.ones(3) * np.deg2rad(degrees) / np.sqrt(3)
                current[:3, :3] = (
                    geometry.target[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
                )
                self.assertEqual(bool(geometry.metrics(current)["in_target"]), expected)


class PegRewardScaleTests(unittest.TestCase):
    def test_position_gaussian_matches_world_distance(self):
        geometry = PegInsertionGeometry(
            dense_reward_type="position_gaussian", dense_reward_scale=0.005
        )
        poses = np.repeat(geometry.target[None], 4, axis=0)
        offsets = np.array(
            [[0, 0, 0], [0.03, 0.04, 0], [0, 0, 0.1], [0, 0, 0.02]]
        )
        poses[:, :3, 3] += offsets
        actual = geometry.metrics(poses)
        np.testing.assert_allclose(
            actual["dense_reward"], 0.005 * np.exp(-500 * (offsets**2).sum(axis=1))
        )
        legacy = PegInsertionGeometry(dense_reward_scale=0.005).metrics(poses)
        for key in ("xy_error", "z_error", "angle_error", "in_target"):
            np.testing.assert_array_equal(actual[key], legacy[key])
        self.assertAlmostEqual(
            geometry.metrics(poses[1])["dense_reward"], actual["dense_reward"][1]
        )

    def test_default_reward_formula_unchanged(self):
        geometry = PegInsertionGeometry()
        metrics = geometry.metrics(geometry.reset_pose(np.random.default_rng(42)))
        expected = 0.05 * np.exp(
            -metrics["xy_error"] / 0.01 - metrics["angle_error"] / 0.1
        )
        expected *= (1 + 2 * np.exp(-metrics["z_error"] / 0.05)) / 3
        self.assertAlmostEqual(metrics["dense_reward"], expected)

    def test_invalid_reward_type_is_rejected(self):
        with self.assertRaises(ValueError):
            PegInsertionGeometry(dense_reward_type="typo")

    def test_zero_scale_preserves_success_geometry(self):
        dense = PegInsertionGeometry()
        sparse = PegInsertionGeometry(dense_reward_scale=0.0)
        poses = np.stack([dense.target, dense.reset_pose(np.random.default_rng(0))])
        dense_metrics = dense.metrics(poses)
        sparse_metrics = sparse.metrics(poses)
        np.testing.assert_array_equal(sparse_metrics["dense_reward"], [0.0, 0.0])
        np.testing.assert_array_equal(sparse_metrics["in_target"], [True, False])
        for key in ("xy_error", "z_error", "angle_error", "in_target"):
            np.testing.assert_array_equal(sparse_metrics[key], dense_metrics[key])
        self.assertAlmostEqual(dense_metrics["dense_reward"][0], 0.05)

    def test_invalid_scale_is_rejected(self):
        for scale in (-0.01, 1.0, 1.1, np.nan, np.inf, -np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                PegInsertionGeometry(dense_reward_scale=scale)


if __name__ == "__main__":
    unittest.main()
