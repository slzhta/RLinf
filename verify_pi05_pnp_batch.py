#!/usr/bin/env python3
"""Build the registered OpenPI loader and inspect exactly one PnP SFT batch."""

from __future__ import annotations

import argparse
import os

import numpy as np


def describe(name, value, indent="") -> None:
    if isinstance(value, dict):
        print(f"{indent}{name}: dict")
        for key, item in value.items():
            describe(str(key), item, indent + "  ")
        return
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    print(f"{indent}{name}: shape={shape}, dtype={dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/home/ubuntu/wangyinghan/wangyitao/pnp_lerobot_dataset",
    )
    parser.add_argument("--model-path", default="/mnt/RLinf/pi05_base_pytorch")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    os.environ["HF_LEROBOT_HOME"] = args.dataset_root

    import openpi.training.data_loader as openpi_data_loader
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(
        "pi05_pnp",
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    loader = openpi_data_loader.create_data_loader(
        config, framework="pytorch", shuffle=False
    )
    observation, actions = next(iter(loader))
    describe("observation", observation)
    describe("actions", actions)

    action_array = np.asarray(actions.cpu() if hasattr(actions, "cpu") else actions)
    if not np.isfinite(action_array).all():
        raise ValueError("processed actions contain NaN/Inf")
    print("batch verification passed")


if __name__ == "__main__":
    main()
