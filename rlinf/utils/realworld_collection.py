"""In-memory capture of executed transitions, committed only by human choice."""

import json
import os
import time
from pathlib import Path

import numpy as np


def numpy_copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


class EpisodeCapture:
    def __init__(self):
        self.observations = {}
        self.observation_times = []
        self.transitions = []

    def observation(self, obs):
        values = {
            key: numpy_copy(obs[key])
            for key in (
                "main_images",
                "wrist_images",
                "extra_view_images",
                "states",
                "task_descriptions",
            )
            if obs.get(key) is not None
        }
        if "main_images" not in values or "states" not in values:
            raise ValueError("Collection requires native images and states")
        if self.observations and values.keys() != self.observations.keys():
            raise ValueError("Observation fields changed within an episode")
        for key, value in values.items():
            if value.shape[0] != 1 or value.dtype.hasobject:
                raise ValueError(f"Unsupported observation field: {key}")
            self.observations.setdefault(key, []).append(value[0].copy())
        self.observation_times.append(time.time())

    def transition(self, action, reward, terminated, truncated, next_obs, started):
        self.transitions.append(
            {
                "actions": numpy_copy(action).reshape(-1),
                "rewards": float(numpy_copy(reward).reshape(-1)[0]),
                "terminated": bool(numpy_copy(terminated).reshape(-1)[0]),
                "truncated": bool(numpy_copy(truncated).reshape(-1)[0]),
                "action_started_at": started,
                "step_finished_at": time.time(),
            }
        )
        self.observation(next_obs)

    def arrays(self, record):
        if not record["completed"] or not self.transitions:
            raise ValueError("Incomplete/aborted episodes cannot be accepted")
        count = len(self.transitions)
        if count != record["steps"] or any(
            len(v) != count + 1 for v in self.observations.values()
        ):
            raise ValueError("Expected T actions and T+1 native observations")
        arrays = {f"obs_{k}": np.stack(v) for k, v in self.observations.items()}
        arrays.update(
            {
                k: np.asarray([t[k] for t in self.transitions])
                for k in self.transitions[0]
            }
        )
        arrays["observation_times"] = np.asarray(self.observation_times)
        if not arrays["terminated"][-1] and not arrays["truncated"][-1]:
            arrays["truncated"][-1] = True
        if not np.isfinite(arrays["actions"]).all():
            raise ValueError("Nonfinite recorded actions")
        return arrays

    def save(self, directory, record, config):
        arrays = self.arrays(record)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        # The metadata marker is written last; interrupted partial saves are not datasets.
        with (directory / "trajectory.npz").open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            "schema": "rlinf_realworld_eval_capture_v1",
            "accepted": True,
            "episode": record,
            "config": config,
            "action_semantics": "composed policy action passed to env.chunk_step, before environment scale/safety clipping",
            "observation_semantics": "native preprocessed environment observations, T+1 including actual terminal observation",
            "observation_keys": list(self.observations),
        }
        with (directory / "metadata.json").open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        return str(directory)


def ask_disposition(record, read=input):
    """Require an explicit choice; EOF stops rather than starting another reset."""
    while True:
        answer = (
            read(
                f"Episode {record['episode_id']}: success={record['success']}, "
                f"steps={record['steps']}. [k] keep / [d] discard / "
                "[q] discard and quit: "
            )
            .strip()
            .lower()
        )
        if answer in ("k", "d", "q"):
            return answer
