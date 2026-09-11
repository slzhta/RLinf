# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict
from torch.distributed.tensor import DTensor


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        return result.stdout if result.returncode == 0 else result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"Unavailable: {exc}"


class FirstPPOUpdateSnapshot:
    """Record one single-actor CNN PPO update without extra model forwards."""

    def __init__(self, actor: Any, shuffle_id: torch.Tensor):
        if int(actor._world_size) != 1 or actor.cfg.actor.model.model_type != "cnn_policy":
            raise ValueError("debug_first_update supports only CNN PPO with one actor rank.")

        batch = actor.rollout_batch
        t_dim, b_dim = batch["prev_logprobs"].shape[:2]
        batch_size = int(actor.cfg.actor.global_batch_size)
        if (t_dim * b_dim) % batch_size:
            raise ValueError("Snapshot batch must be divisible by actor.global_batch_size.")
        self.steps_per_epoch = t_dim * b_dim // batch_size
        self.total_steps = self.steps_per_epoch * int(actor.cfg.algorithm.update_epoch)
        warmup = int(actor.critic_warmup_steps)
        self.checkpoint_steps = {
            step
            for step in (warmup, warmup + 1, self.steps_per_epoch, self.total_steps)
            if 0 < step <= self.total_steps
        }

        domain_metrics = {}
        if actor.cfg.algorithm.get("sim_real_rl_co_training", False):
            if not actor._use_keyed_actor_trajectory():
                raise ValueError("Co-training snapshots require the domain-buffer path.")
            domain_metrics = dict(actor._co_training_buffer_metrics)
            real_count = int(domain_metrics["buffer/train_real_rollouts"])
            sim_count = int(domain_metrics["buffer/train_sim_rollouts"])
            if real_count + sim_count != b_dim:
                raise ValueError("Snapshot domain counts do not match the training batch.")
            # _recv_domain_buffered_batch concatenates real blocks before sim blocks.
            domains = ["real"] * real_count + ["sim"] * sim_count
        else:
            domain = "real" if actor.cfg.env.train.env_type == "realworld" else "sim"
            domains = [domain] * b_dim

        options = actor.cfg.actor.debug_first_update
        output_dir = options.get("output_dir", None)
        self.directory = (
            Path(output_dir).expanduser()
            if output_dir
            else Path(actor.cfg.runner.logger.log_path) / "debug_first_update"
        ).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "format_version": 1,
            "status": "capturing",
            "created_at_unix": time.time(),
            "rank": int(actor._rank),
            "world_size": int(actor._world_size),
            "time_steps": t_dim,
            "trajectory_blocks": b_dim,
            "steps_per_epoch": self.steps_per_epoch,
            "total_optimizer_steps": self.total_steps,
            "checkpoint_steps": sorted(self.checkpoint_steps),
            "saved_optimizer_steps": [],
        }
        self._write_manifest()
        self._save(
            "batch.pt",
            {
                "rollout_batch": batch,
                "shuffle_id": shuffle_id,
                "trajectory_domains": domains,
                "trajectory_indices": torch.arange(b_dim),
                "buffer_metrics": domain_metrics,
            },
        )
        self._save_state(actor, "before_update.pt")
        self._save_provenance(actor)

    def _save(self, filename: str, payload: Any) -> None:
        path = self.directory / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as stream:
            torch.save(_cpu_copy(payload), stream)
        temporary.rename(path)

    def _write_manifest(self) -> None:
        temporary = self.directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "manifest.json")

    def _save_state(self, actor: Any, filename: str) -> None:
        rng = _rng_state()
        parameters = dict(actor.model.named_parameters())
        names_by_id = {id(param): name for name, param in parameters.items()}
        model_state, optimizer_full = get_state_dict(
            model=actor.model,
            optimizers=actor.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        self._save(
            filename,
            {
                "model": model_state,
                "optimizer_full": optimizer_full,
                "optimizer": actor.optimizer.state_dict(),
                "optimizer_parameter_names": [
                    [names_by_id[id(param)] for param in group["params"]]
                    for group in actor.optimizer.param_groups
                ],
                "scheduler": actor.lr_scheduler.state_dict(),
                "grad_scaler": actor.grad_scaler.state_dict(),
                "optimizer_steps": int(actor.optimizer_steps),
                "critic_warmup_steps": int(actor.critic_warmup_steps),
                "store_requires_grad_param_name": list(
                    actor.store_requires_grad_param_name
                ),
                "requires_grad": {
                    name: param.requires_grad for name, param in parameters.items()
                },
                "module_training": {
                    name: module.training
                    for name, module in actor.model.named_modules()
                },
                "version": int(actor.version),
                "rng": rng,
            },
        )

    def _save_provenance(self, actor: Any) -> None:
        root = Path(__file__).resolve().parents[2]
        sources = {}
        for relative in (
            "rlinf/utils/ppo_update_snapshot.py",
            "rlinf/workers/actor/async_ppo_fsdp_worker.py",
            "rlinf/workers/actor/fsdp_actor_worker.py",
            "rlinf/runners/async_ppo_embodied_runner.py",
            "rlinf/hybrid_engines/fsdp/fsdp_model_manager.py",
            "rlinf/algorithms/losses.py",
            "rlinf/algorithms/advantages.py",
            "rlinf/models/embodiment/cnn_policy/cnn_policy.py",
            "rlinf/models/embodiment/modules/resnet_utils.py",
        ):
            sources[relative] = (root / relative).read_text(encoding="utf-8")
        self._save(
            "provenance.pt",
            {
                "config": OmegaConf.to_container(actor.cfg, resolve=True),
                "git_head": _git_output(root, "rev-parse", "HEAD").strip(),
                "git_diff": _git_output(root, "diff", "HEAD", "--", "rlinf"),
                "sources": sources,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "hostname": platform.node(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
        )

    def save_step(self, actor: Any) -> None:
        """Save state after selected optimizer steps, including warmup transition."""
        step = int(actor.optimizer_steps)
        if step not in self.checkpoint_steps:
            return
        self._save_state(actor, f"after_optimizer_step_{step:04d}.pt")
        self.manifest["saved_optimizer_steps"].append(step)
        self._write_manifest()

    def finish(self, actor: Any, metrics: dict, mean_metrics: dict) -> None:
        """Mark completion only after all update artifacts have been written."""
        if (
            int(actor.optimizer_steps) != self.total_steps
            or set(self.manifest["saved_optimizer_steps"]) != self.checkpoint_steps
        ):
            raise RuntimeError("First PPO update snapshot is missing optimizer steps.")
        self._save(
            "update_result.pt",
            {
                "metrics": metrics,
                "mean_metrics": mean_metrics,
                "scheduler": actor.lr_scheduler.state_dict(),
                "grad_scaler": actor.grad_scaler.state_dict(),
                "rng": _rng_state(),
            },
        )
        self.manifest["status"] = "complete"
        self.manifest["completed_at_unix"] = time.time()
        self._write_manifest()
