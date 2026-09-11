#!/usr/bin/env python3
"""Validate raw NPZ demonstrations produced by collect_pnp_motionplanning.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    files = sorted(args.dataset.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"no episode_*.npz in {args.dataset}")
    all_actions, lengths = [], []
    for path in files:
        with np.load(path) as ep:
            t = len(ep["actions"])
            expected = {
                "base_images": (t, None, None, 3), "wrist_images": (t, None, None, 3),
                "states": (t, 14), "actions": (t, 7), "rewards": (t,),
                "terminated": (t,), "truncated": (t,),
            }
            for key, shape in expected.items():
                actual = ep[key].shape
                if len(actual) != len(shape) or any(w is not None and a != w for a, w in zip(actual, shape)):
                    raise ValueError(f"{path.name}: {key} shape {actual}, expected {shape}")
                if not np.all(np.isfinite(ep[key])):
                    raise ValueError(f"{path.name}: non-finite values in {key}")
            if ep["base_images"].dtype != np.uint8 or ep["wrist_images"].dtype != np.uint8:
                raise ValueError(f"{path.name}: images must be uint8")
            if np.max(np.abs(ep["actions"])) > 1.00001:
                raise ValueError(f"{path.name}: action outside [-1,1]")
            grip = ep["actions"][:, 6]
            if not np.all(np.isin(grip, [-1.0, 1.0])):
                raise ValueError(f"{path.name}: gripper labels are not persistent +/-1")
            transitions = np.flatnonzero(grip[1:] != grip[:-1]) + 1
            if len(transitions) != 2 or grip[0] != 1 or grip[-1] != 1:
                raise ValueError(f"{path.name}: expected gripper sequence open-close-open")
            all_actions.append(ep["actions"])
            lengths.append(t)
    actions = np.concatenate(all_actions)
    print(f"episodes={len(files)} frames={len(actions)} length[min/mean/max]={min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
    print("action min:", actions.min(axis=0))
    print("action max:", actions.max(axis=0))
    print("action mean:", actions.mean(axis=0))
    print("action std:", actions.std(axis=0))


if __name__ == "__main__":
    main()