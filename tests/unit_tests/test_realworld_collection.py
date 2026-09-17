"""Pure array tests: no environment, Ray, GPU, or robot initialization."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rlinf.utils.realworld_collection import EpisodeCapture, ask_disposition


class CollectionTests(unittest.TestCase):
    def capture(self):
        c = EpisodeCapture()
        obs = {
            "main_images": np.zeros((1, 4, 4, 3), dtype=np.uint8),
            "states": np.zeros((1, 6), dtype=np.float32),
            "task_descriptions": ["Insert the peg"],
        }
        c.observation(obs)
        obs["states"][:] = 1
        c.transition(np.full((1, 1, 6), 0.25), [1], [True], [False], obs, 1.0)
        return c

    def test_native_alignment_and_language(self):
        c = self.capture()
        a = c.arrays({"completed": True, "steps": 1})
        np.testing.assert_array_equal(a["obs_states"][:, 0], [0, 1])
        self.assertEqual(a["actions"].shape, (1, 6))
        self.assertEqual(a["obs_main_images"].shape, (2, 4, 4, 3))
        self.assertEqual(a["obs_task_descriptions"].tolist(), ["Insert the peg"] * 2)
        self.assertTrue(a["terminated"][-1])
        self.assertFalse(a["truncated"][-1])

    def test_saved_arrays_need_no_pickle_and_refuse_overwrite(self):
        c = self.capture()
        record = {"completed": True, "steps": 1, "success": True}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "episode_000001"
            c.save(p, record, {"task": "peg"})
            with np.load(p / "trajectory.npz", allow_pickle=False) as data:
                self.assertEqual(len(data["obs_states"]), len(data["actions"]) + 1)
            self.assertTrue(json.loads((p / "metadata.json").read_text())["accepted"])
            with self.assertRaises(FileExistsError):
                c.save(p, record, {})

    def test_incomplete_not_accepted(self):
        with self.assertRaises(ValueError):
            self.capture().arrays({"completed": False, "steps": 1})

    def test_driver_limit_is_truncation_not_success(self):
        c = self.capture()
        c.transitions[-1]["terminated"] = False
        a = c.arrays({"completed": True, "steps": 1})
        self.assertTrue(a["truncated"][-1])
        self.assertFalse(a["terminated"][-1])

    def test_explicit_review_no_default_keep(self):
        answers = iter(["", "invalid", "k"])
        record = {"episode_id": 1, "success": True, "steps": 1}
        self.assertEqual(ask_disposition(record, lambda _: next(answers)), "k")
        self.assertEqual(ask_disposition(record, lambda _: "q"), "q")

        def eof(_):
            raise EOFError

        with self.assertRaises(EOFError):
            ask_disposition(record, eof)


if __name__ == "__main__":
    unittest.main()
