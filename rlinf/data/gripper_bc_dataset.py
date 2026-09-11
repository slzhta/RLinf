# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read aligned gripper demonstrations from NPZ or embedded-image LeRobot v2."""

from bisect import bisect_right
from collections import OrderedDict
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


class GripperBCDataset(Dataset):
    """Bounded episode cache; callers shuffle episodes and frames within episodes."""

    def __init__(
        self,
        files: list[Path],
        image_keys: list[str],
        state_dim: int,
        state_key: str = "states",
        action_key: str = "actions",
        action_encoding: str = "signed",
        cache_size: int = 2,
    ) -> None:
        if not files or not image_keys or cache_size < 1:
            raise ValueError(
                "BC needs episodes, at least one camera and a positive cache size."
            )
        if action_encoding not in ("signed", "zero_one"):
            raise ValueError(
                "action_encoding must be signed or zero_one, with positive/one=open."
            )
        self.files = files
        self.image_keys = image_keys
        self.state_dim = state_dim
        self.state_key = state_key
        self.action_key = action_key
        self.action_encoding = action_encoding
        self.cache_size = cache_size
        self.cache: OrderedDict[int, dict] = OrderedDict()
        self.offsets = [0]
        for path in files:
            with self._open_episode(path, load_images=False) as data:
                labels = self._labels(data[action_key])
                if not len(labels):
                    raise ValueError(f"Empty demonstration: {path}")
                for key in image_keys + ([state_key] if state_dim else []):
                    if key not in data:
                        raise ValueError(f"Missing {key} in {path}")
            self.offsets.append(self.offsets[-1] + len(labels))

    @contextmanager
    def _open_episode(self, path: Path, load_images: bool) -> Iterator[dict]:
        """Decode raw commands without any SFT action normalization or shifting."""
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                yield data
            return
        if path.suffix != ".parquet":
            raise ValueError(f"Unsupported BC episode format: {path}")
        import pyarrow.parquet as pq
        from PIL import Image

        available = pq.read_schema(path).names
        required = [self.action_key, "episode_index", *self.image_keys]
        if self.state_dim:
            required.append(self.state_key)
        missing = set(required) - set(available)
        if missing:
            raise ValueError(f"Missing LeRobot fields {sorted(missing)} in {path}")
        columns = required if load_images else [self.action_key, "episode_index"]
        table = pq.read_table(path, columns=columns)
        if len(set(table["episode_index"].to_pylist())) != 1:
            raise ValueError("BC requires one episode per parquet file (LeRobot v2).")
        # Retain schema keys for the cheap constructor field check.
        arrays = dict.fromkeys(available)
        arrays[self.action_key] = np.asarray(
            table[self.action_key].to_pylist(), dtype=np.float32
        )
        if load_images:
            if self.state_dim:
                arrays[self.state_key] = np.asarray(
                    table[self.state_key].to_pylist(), dtype=np.float32
                )
            for key in self.image_keys:
                frames = []
                for record in table[key].to_pylist():
                    if not isinstance(record, dict) or not record.get("bytes"):
                        raise ValueError(
                            "LeRobot BC needs embedded image bytes; external images/videos are unsupported."
                        )
                    with Image.open(BytesIO(record["bytes"])) as image:
                        frames.append(np.asarray(image.convert("RGB")))
                arrays[key] = np.stack(frames)
        yield arrays

    def _labels(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(
                "BC actions must be aligned [T,7] commands, not action chunks."
            )
        commands = actions[:, 6]
        low = -1.0 if self.action_encoding == "signed" else 0.0
        if not np.all(np.isclose(commands, low) | np.isclose(commands, 1.0)):
            raise ValueError(
                "BC gripper labels must be binary commands in the declared encoding."
            )
        return np.isclose(commands, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return self.offsets[-1]

    def episode_order(self, seed: int) -> list[int]:
        """Shuffle without repeatedly decompressing unrelated full episodes."""
        rng = np.random.default_rng(seed)
        result = []
        for episode in rng.permutation(len(self.files)):
            result.extend(
                rng.permutation(
                    np.arange(self.offsets[episode], self.offsets[episode + 1])
                ).tolist()
            )
        return result

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.offsets, index) - 1
        if episode not in self.cache:
            with self._open_episode(self.files[episode], load_images=True) as data:
                labels = self._labels(data[self.action_key])
                arrays = {key: data[key] for key in self.image_keys}
                for key, images in arrays.items():
                    if (
                        images.ndim != 4
                        or images.shape[0] != len(labels)
                        or images.shape[-1] != 3
                        or images.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"{key} must contain T aligned uint8 NHWC images."
                        )
                if self.state_dim:
                    states = data[self.state_key]
                    if (
                        states.shape != (len(labels), self.state_dim)
                        or not np.isfinite(states).all()
                    ):
                        raise ValueError(
                            "BC state dimensions/alignment must match the gripper model."
                        )
                    arrays[self.state_key] = states
                transition = np.zeros(len(labels), dtype=bool)
                for t in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
                    transition[max(0, t - 2) : min(len(labels), t + 3)] = True
                arrays["_labels"] = labels
                arrays["_transition"] = transition
                self.cache[episode] = arrays
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        arrays = self.cache[episode]
        frame = index - self.offsets[episode]
        obs = {
            "main_images": torch.from_numpy(arrays[self.image_keys[0]][frame].copy())
        }
        if len(self.image_keys) > 1:
            obs["extra_view_images"] = torch.from_numpy(
                np.stack([arrays[key][frame] for key in self.image_keys[1:]])
            )
        if self.state_dim:
            obs["states"] = torch.from_numpy(
                arrays[self.state_key][frame].copy()
            ).float()
        return (
            obs,
            torch.tensor([arrays["_labels"][frame]]),
            torch.tensor([bool(arrays["_transition"][frame])]),
        )
