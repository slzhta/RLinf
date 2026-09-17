"""No hardware, Ray, or GPU needed for episode accounting tests."""

import ast
import asyncio
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from rlinf.utils.realworld_eval import (
    evaluation_checkpoint,
    evaluation_observation,
    make_peg_z_guard,
    run_episode,
    summarize,
)


class ResidualEvalTests(unittest.TestCase):
    def config(self, model="residual_policy", checkpoint=None, allow=True):
        return OmegaConf.create(
            {
                "runner": {"ckpt_path": checkpoint},
                "actor": {"model": {"model_type": model}},
                "rollout": {"residual_base_inference": True},
                "evaluation": {"allow_initial_residual": allow},
            }
        )

    def test_explicit_initial_residual(self):
        self.assertIsNone(evaluation_checkpoint(self.config()))

    def test_missing_path_never_falls_back_to_initial(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                evaluation_checkpoint(
                    self.config(checkpoint=str(Path(directory) / "missing.pt"))
                )

    def test_cnn_still_requires_checkpoint(self):
        with self.assertRaises(ValueError):
            evaluation_checkpoint(self.config(model="cnn_policy"))
        with self.assertRaises(ValueError):
            evaluation_checkpoint(self.config(allow=False))

    def test_existing_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            path.touch()
            self.assertEqual(
                evaluation_checkpoint(self.config(checkpoint=str(path))), str(path)
            )

    def test_reset_mask_and_no_mutation(self):
        obs = {"main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8)}
        for first in [True, False, False, True]:
            result = evaluation_observation(obs, True, first)
            self.assertEqual(result["_residual_reset_mask"].tolist(), [first])
            self.assertEqual(result["_residual_reset_mask"].dtype, np.bool_)
        self.assertNotIn("_residual_reset_mask", obs)
        self.assertIs(evaluation_observation(obs, False, True), obs)


class PegZGuardTests(unittest.TestCase):
    def config(self):
        return OmegaConf.create(
            {
                "evaluation": {"stop_at_target_z_distance": 0.01},
                "runner": {"only_eval": True},
                "actor": {
                    "model": {"model_type": "cnn_policy", "num_action_chunks": 1}
                },
                "algorithm": {"loss_type": "embodied_sac"},
                "env": {
                    "eval": {
                        "init_params": {"id": "FrankaCoTrainingPegInsertionEnv-v1"},
                        "state_keys": ["ee_target_delta"],
                        "auto_reset": False,
                        "total_num_envs": 1,
                        "override_cfg": {
                            "peg_config": {"target_ee_pose": [0.6, 0.1, 0, 0, 0, 0]}
                        },
                    }
                },
            }
        )

    def obs(self, z):
        return {"states": np.array([[0, 0, z, 0, 0, 0]], dtype=np.float32)}

    def test_disabled_by_default(self):
        self.assertIsNone(make_peg_z_guard(OmegaConf.create({"evaluation": {}})))

    def test_boundary_and_both_sides(self):
        guard = make_peg_z_guard(self.config())
        for z in (0, 0.009, -0.009, 0.01, -0.01):
            self.assertIsNotNone(guard(self.obs(z)))
        self.assertIsNone(guard(self.obs(0.0101)))
        self.assertIsNone(guard(self.obs(-0.0101)))

    def test_target_frame_is_rotated_back_to_base(self):
        cfg = self.config()
        cfg.env.eval.override_cfg.peg_config.target_ee_pose[4] = np.pi / 2
        guard = make_peg_z_guard(cfg)
        self.assertIsNotNone(guard(self.obs(0.1)))
        obs = self.obs(0)
        obs["states"][0, 0] = 0.1
        self.assertIsNone(guard(obs))

    def test_invalid_state_and_config_fail_closed(self):
        guard = make_peg_z_guard(self.config())
        for obs in ({}, self.obs(float("nan")), {"states": np.zeros((2, 6))}):
            with self.assertRaises((ValueError, KeyError)):
                guard(obs)
        cfg = self.config()
        cfg.runner.only_eval = False
        with self.assertRaises(ValueError):
            make_peg_z_guard(cfg)

    def test_stop_before_next_prediction_and_keep_environment_success(self):
        for env_success in (False, True):
            calls = []

            def step(action):
                calls.append("step")
                return (
                    self.obs(0.009),
                    float(env_success),
                    env_success,
                    False,
                    {"success": env_success},
                )

            def predict(obs):
                calls.append("predict")
                return 0

            result = run_episode(
                lambda: (self.obs(0.1), {}),
                step,
                predict,
                240,
                stop_guard=make_peg_z_guard(self.config()),
            )
            self.assertEqual(calls, ["predict", "step"])
            self.assertEqual(result["status"], "z_guard")
            self.assertEqual(result["success"], env_success)
            self.assertTrue(result["completed"])
            self.assertTrue(summarize([result], 1)["manual_scoring_required"])

    def test_initial_guard_sends_no_action(self):
        def forbidden(*args):
            self.fail("No action should be issued at target height")

        result = run_episode(
            lambda: (self.obs(0), {}),
            forbidden,
            forbidden,
            240,
            stop_guard=make_peg_z_guard(self.config()),
        )
        self.assertEqual(result["steps"], 0)

    def test_worker_resets_once_per_guard_including_last_episode(self):
        path = Path(__file__).parents[2] / "rlinf/workers/env/realworld_eval_worker.py"
        # Execute only the episode method with fake channels/environment, never Ray.
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_episode"
        )
        method.name = "worker_run_episode"
        scope = {
            "asyncio": asyncio,
            "os": os,
            "time": time,
            "Path": Path,
            "np": np,
            "RecordVideo": type("UnusedVideo", (), {}),
            "append_record": lambda *args: None,
            "make_peg_z_guard": make_peg_z_guard,
            "run_episode": run_episode,
            "evaluation_observation": evaluation_observation,
            "prepare_actions": lambda **kw: kw["raw_chunk_actions"],
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            scope,
        )
        calls = []

        def reset():
            calls.append("reset")
            return self.obs(0.1), {}

        def step(action):
            calls.append("step")
            return [self.obs(0.009)], 0.0, False, False, [{"success": False}]

        cfg = self.config()
        cfg.actor.model.action_dim = 6
        cfg.rollout = {}
        cfg.env.eval.env_type = "realworld"
        cfg.env.eval.max_episode_steps = 240
        cfg.evaluation.run_relative_dir = "mock"
        worker = types.SimpleNamespace(
            cfg=cfg,
            eval_env_list=[types.SimpleNamespace(reset=reset, chunk_step=step)],
            begin_rollout_video=lambda **kw: None,
        )

        def empty(**kwargs):
            raise asyncio.QueueEmpty

        control = types.SimpleNamespace(get_nowait=empty)
        observations = types.SimpleNamespace(put=lambda *args, **kw: None)
        actions = types.SimpleNamespace(
            get_nowait=lambda **kw: {"actions": np.zeros((1, 1, 6))}
        )
        with patch.dict(os.environ, {"RLINF_LOG_PATH": "/unused"}):
            for episode in (1, 2):
                result = scope["worker_run_episode"](
                    worker, episode, observations, actions, control
                )
                self.assertTrue(result["guard_reset_completed"])
                self.assertEqual(result["status"], "z_guard")
        self.assertEqual(calls, ["reset", "step", "reset", "step", "reset"])


class EpisodeTests(unittest.TestCase):
    def execute(self, outcome, terminal_at=2, limit=5):
        self.resets = 0
        self.steps = 0

        def reset():
            self.resets += 1
            self.steps = 0
            return {}, {}

        def step(action):
            self.steps += 1
            terminal = self.steps == terminal_at
            return (
                {},
                float(terminal and outcome == "success"),
                terminal,
                False,
                {
                    "episode": {
                        "success_once": terminal and outcome == "success",
                        "aborted": terminal and outcome == "aborted",
                    }
                },
            )

        return run_episode(reset, step, lambda _: 0, limit)

    def test_terminal_stops_without_extra_action_or_reset(self):
        result = self.execute("success")
        self.assertEqual((self.resets, self.steps), (1, 2))
        self.assertTrue(result["success"])

    def test_failure_is_completed(self):
        result = self.execute("failure")
        self.assertEqual(result["status"], "failure")
        self.assertTrue(result["completed"])

    def test_timeout_counts_failure(self):
        result = self.execute("failure", terminal_at=100)
        self.assertEqual((result["status"], result["steps"]), ("timeout", 5))

    def test_abort_separate_from_completed(self):
        records = [self.execute("success"), self.execute("aborted")]
        report = summarize(records, 40)
        self.assertEqual((report["completed"], report["interrupted"]), (1, 1))
        self.assertEqual(report["successes_per_attempt"], 0.5)
        self.assertFalse(report["finished"])

    def test_forty_episodes(self):
        records = [self.execute("success" if i % 2 else "failure") for i in range(40)]
        report = summarize(records, 40)
        self.assertTrue(report["finished"])
        self.assertEqual((report["completed"], report["success_rate"]), (40, 0.5))


if __name__ == "__main__":
    unittest.main()
