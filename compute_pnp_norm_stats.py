#!/usr/bin/env python3
"""Compute OpenPI normalization statistics for RLinf's ``pi05_pnp`` config."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tqdm
import tyro


def create_torch_dataloader(
    data_config,
    action_horizon: int,
    batch_size: int,
    model_config,
    num_workers: int,
    max_frames: int | None = None,
):
    # Import only after main() sets HF_LEROBOT_HOME. The installed LeRobot
    # version resolves and caches its local dataset root at import time.
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as transforms

    class RemoveStrings(transforms.DataTransformFn):
        def __call__(self, value: dict) -> dict:
            return {
                key: item
                for key, item in value.items()
                if not np.issubdtype(np.asarray(item).dtype, np.str_)
            }

    if data_config.repo_id is None:
        raise ValueError("data config must have a repo_id")

    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max(1, max_frames // batch_size)
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    if num_batches < 1:
        raise ValueError(
            f"dataset has {len(dataset)} frames, smaller than batch size {batch_size}"
        )

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return loader, num_batches


def main(
    dataset_root: str = (
        "/home/ubuntu/wangyinghan/wangyitao/pnp_lerobot_dataset"
    ),
    model_path: str = "/mnt/RLinf/lerobot_pi05_base",
    config_name: str = "pi05_pnp",
    batch_size: int = 32,
    num_workers: int = 4,
    max_frames: int | None = None,
) -> None:
    """Compute stats after PnP repacking/padding and save them as OpenPI assets."""
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root_path)

    # These imports must stay below the environment assignment above.
    import openpi.shared.normalize as normalize

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config(
        config_name,
        model_path=model_path,
        batch_size=batch_size,
    )
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("resolved data config has no repo_id")

    dataset_path = dataset_root_path / data_config.repo_id
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"dataset not found: {dataset_path}. dataset_root must be the parent "
            "directory and repo_id must be the dataset directory name."
        )

    # This PnP dataset is LeRobot/parquet, not RLDS.
    loader, num_batches = create_torch_dataloader(
        data_config=data_config,
        action_horizon=config.model.action_horizon,
        batch_size=batch_size,
        model_config=config.model,
        num_workers=num_workers,
        max_frames=max_frames,
    )

    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing PnP stats"):
        for key in stats:
            value = np.asarray(batch[key])
            if not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN or Inf after transforms")
            stats[key].update(value)

    norm_stats = {
        key: running_stats.get_statistics()
        for key, running_stats in stats.items()
    }
    output_path = config.assets_dirs / data_config.repo_id
    print(f"dataset: {dataset_path}")
    print(f"frames used: {num_batches * batch_size}")
    print(f"writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(f"saved: {output_path / 'norm_stats.json'}")


if __name__ == "__main__":
    tyro.cli(main)
