# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline gripper BC without Ray, OpenPI or a simulator; launch with python -m."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rlinf.data.gripper_bc_dataset import GripperBCDataset
from rlinf.models.embodiment.residual_policy.gripper_cnn import (
    GripperCNN,
    GripperCNNConfig,
)
from rlinf.utils.logging import get_logger


def training_replica(
    model: GripperCNN, device: torch.device, data_parallel: bool
) -> torch.nn.Module:
    """Split one global BC batch across visible GPUs, keeping portable weights."""
    if not data_parallel:
        return model
    if device.type != "cuda" or device.index not in (None, 0):
        raise ValueError(
            "DataParallel requires --device cuda:0 and CUDA_VISIBLE_DEVICES."
        )
    if torch.cuda.device_count() < 2:
        raise ValueError("DataParallel requires at least two visible CUDA GPUs.")
    return torch.nn.DataParallel(
        model, device_ids=list(range(torch.cuda.device_count()))
    )


def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Measure calibration and both command classes on held-out episodes."""
    model.eval()
    totals = {
        "n": 0,
        "loss": 0.0,
        "correct": 0,
        "open_n": 0,
        "open_ok": 0,
        "close_n": 0,
        "close_ok": 0,
        "switch_n": 0,
        "switch_ok": 0,
    }
    with torch.no_grad():
        for obs, target, switch in loader:
            obs = {key: value.to(device) for key, value in obs.items()}
            target = target.to(device)
            switch = switch.to(device)
            logits = model(obs)
            correct = (logits >= 0) == target.bool()
            totals["n"] += target.numel()
            totals["loss"] += F.binary_cross_entropy_with_logits(
                logits, target, reduction="sum"
            ).item()
            totals["correct"] += correct.sum().item()
            for name, mask in (
                ("open", target.bool()),
                ("close", ~target.bool()),
                ("switch", switch),
            ):
                totals[name + "_n"] += mask.sum().item()
                totals[name + "_ok"] += (correct & mask).sum().item()
    metrics = {
        "bce": totals["loss"] / totals["n"],
        "accuracy": totals["correct"] / totals["n"],
    }
    for name in ("open", "close", "switch"):
        metrics[name + "_count"] = totals[name + "_n"]
        metrics[name + "_accuracy"] = (
            totals[name + "_ok"] / totals[name + "_n"] if totals[name + "_n"] else 0.0
        )
    if not totals["open_n"] or not totals["close_n"]:
        raise ValueError(
            "Validation episodes must contain both open and close commands."
        )
    metrics["balanced_accuracy"] = (
        metrics["open_accuracy"] + metrics["close_accuracy"]
    ) / 2
    return metrics


def main() -> None:
    """Train only the independent gripper CNN and export BC initialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-format", choices=["auto", "npz", "lerobot"], default="auto"
    )
    parser.add_argument("--image-keys", nargs="+")
    parser.add_argument("--state-key")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument(
        "--action-encoding", choices=["signed", "zero_one"], default="signed"
    )
    parser.add_argument("--state-dim", type=int, default=14)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Use visible GPUs for one shared model; batch-size is global.",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="0 uses all episodes; nonzero is for smoke tests.",
    )
    parser.add_argument(
        "--max-train-batches", type=int, default=0, help="0 trains complete epochs."
    )
    args = parser.parse_args()
    if (
        not 0 < args.validation_fraction < 1
        or min(args.epochs, args.batch_size, args.cpu_threads) < 1
        or args.lr <= 0
    ):
        parser.error(
            "Use a validation fraction in (0,1) and positive training settings."
        )
    if args.max_episodes < 0 or args.max_train_batches < 0 or args.log_every < 1:
        parser.error("Smoke-test limits must be nonnegative.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if args.data_format == "auto":
        args.data_format = (
            "lerobot" if (args.data_root / "meta/info.json").is_file() else "npz"
        )
    if args.data_format == "lerobot":
        info = json.loads((args.data_root / "meta/info.json").read_text())
        if not str(info.get("codebase_version", "")).startswith("v2.") or info.get(
            "total_videos", 0
        ):
            raise ValueError(
                "Gripper BC supports LeRobot v2 with embedded images, not video/v3 datasets."
            )
        files = sorted(args.data_root.glob("data/*/episode_*.parquet"))
        if len(files) != info["total_episodes"]:
            raise ValueError(
                "LeRobot episode files do not match meta/info.json; dataset may be incomplete."
            )
        args.image_keys = args.image_keys or ["image", "wrist_image"]
        args.state_key = args.state_key or "state"
    else:
        files = sorted(args.data_root.glob("episode_*.npz"))
        args.image_keys = args.image_keys or ["base_images", "wrist_images"]
        args.state_key = args.state_key or "states"
    rng = np.random.default_rng(args.seed)
    rng.shuffle(files)
    if args.max_episodes:
        files = files[: args.max_episodes]
    if len(files) < 2:
        raise ValueError(
            "BC needs at least two episodes for a disjoint validation split."
        )
    val_count = max(
        1, min(len(files) - 1, round(len(files) * args.validation_fraction))
    )
    val_files, train_files = files[:val_count], files[val_count:]

    def dataset(paths: list[Path]) -> GripperBCDataset:
        return GripperBCDataset(
            paths,
            args.image_keys,
            args.state_dim,
            args.state_key,
            args.action_key,
            args.action_encoding,
        )

    train, validation = dataset(train_files), dataset(val_files)
    cfg = GripperCNNConfig(
        len(args.image_keys), args.image_size, args.state_dim, args.hidden_dim
    )
    device = torch.device(args.device)
    model = GripperCNN(cfg).to(device)
    training_model = training_replica(model, device, args.data_parallel)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    manifest = {
        "train_episodes": [str(p) for p in train_files],
        "validation_episodes": [str(p) for p in val_files],
        "config": asdict(cfg),
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger = get_logger()
    start_time = time.monotonic()
    step = 0
    logger.info(
        json.dumps(
            {
                "event": "training_started",
                "train_frames": len(train),
                "validation_frames": len(validation),
                "epochs": args.epochs,
                "global_batch_size": args.batch_size,
                "gpu_count": len(training_model.device_ids)
                if args.data_parallel
                else int(device.type == "cuda"),
            }
        )
    )
    best = -1.0
    for epoch in range(args.epochs):
        training_model.train()
        loader = DataLoader(
            train,
            batch_size=args.batch_size,
            sampler=train.episode_order(args.seed + epoch),
            num_workers=0,
        )
        loss_sum, examples = 0.0, 0
        for batch_index, (obs, target, _) in enumerate(loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            obs = {key: value.to(device) for key, value in obs.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(training_model(obs), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            loss_sum += loss.item() * target.numel()
            examples += target.numel()
            if step % args.log_every == 0:
                logger.info(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": epoch + 1,
                            "step": step,
                            "train_bce": loss_sum / examples,
                            "elapsed_seconds": time.monotonic() - start_time,
                        }
                    )
                )
        metrics = evaluate(
            training_model,
            DataLoader(validation, batch_size=args.batch_size, num_workers=0),
            device,
        )
        record = {
            "epoch": epoch + 1,
            "step": step,
            "train_bce": loss_sum / examples,
            **metrics,
        }
        logger.info(json.dumps(record))
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        model.save_bc(args.output_dir / "last.pt", **record)
        if metrics["balanced_accuracy"] > best:
            best = metrics["balanced_accuracy"]
            model.save_bc(args.output_dir / "best.pt", **record)


if __name__ == "__main__":
    main()
