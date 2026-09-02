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
import os

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.config import SupportedModel, get_supported_model
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _apply_sft_drq(batch: dict) -> None:
    """Augment the visual inputs consumed by the supervised policy forward."""
    forward_inputs = batch.get("forward_inputs")
    if not isinstance(forward_inputs, dict):
        raise ValueError("DAgger DRQ requires a forward_inputs observation dict.")
    drq.apply_drq(forward_inputs, pad=4)


class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))

    def init_worker(self):
        super().setup_model_and_optimizer()
        self.setup_dagger_components()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def setup_dagger_components(self):
        """Initialize DAgger-specific replay buffer state."""
        seed = int(self.cfg.actor.get("seed", 1234)) + int(self._rank)
        replay_cfg = self.cfg.algorithm.replay_buffer
        auto_save_path = replay_cfg.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        load_path = replay_cfg.get("load_path", None)
        self._offline_replay_load_path = load_path
        self._replay_sampling_weight_key = replay_cfg.get("sampling_weight_key", None)
        self._sample_without_replacement = bool(
            replay_cfg.get("sample_without_replacement", False)
        )
        self._offline_data_format_version = None
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=replay_cfg.enable_cache,
            cache_size=replay_cfg.cache_size,
            sample_window_size=replay_cfg.sample_window_size,
            auto_save=replay_cfg.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=replay_cfg.get("trajectory_format", "pt"),
            cache_all_when_not_saving=load_path is None,
        )
        if load_path is not None:
            self._validate_offline_dataset_manifest(load_path)
            self.replay_buffer.load_checkpoint(
                load_path,
                is_distributed=self._world_size > 1,
                local_rank=self._rank,
                world_size=self._world_size,
                restore_seed=False,
            )
            stats = self.replay_buffer.get_stats()
            self.log_on_first_rank(
                "Loaded offline demonstration buffer from "
                f"{load_path}: {stats['num_trajectories']} trajectories, "
                f"{stats['total_samples']} samples, "
                f"{stats['cache_bytes'] / 2**30:.2f} GiB cache"
            )

    def _validate_offline_dataset_manifest(self, load_path: str) -> None:
        """Validate optional provenance constraints for an offline dataset."""
        replay_cfg = self.cfg.algorithm.replay_buffer
        manifest_path = replay_cfg.get("manifest_path", None)
        required_version = replay_cfg.get("required_data_format_version", None)
        required_trajectories = replay_cfg.get("required_num_trajectories", None)
        if manifest_path is None:
            if required_version is not None or required_trajectories is not None:
                raise ValueError("Offline dataset constraints require manifest_path.")
            return
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Offline dataset manifest not found: {manifest_path}"
            )

        with open(manifest_path) as file:
            manifest = json.load(file)
        with open(os.path.join(load_path, "metadata.json")) as file:
            replay_metadata = json.load(file)

        data_version = manifest.get("format_version")
        if required_version is not None and data_version != required_version:
            raise ValueError(
                f"Offline dataset format {data_version!r} does not match required "
                f"version {required_version!r}."
            )
        manifest_trajectories = int(manifest.get("collected_episodes", -1))
        replay_trajectories = int(replay_metadata.get("size", -1))
        expected_trajectories = (
            int(required_trajectories)
            if required_trajectories is not None
            else manifest_trajectories
        )
        if not (manifest_trajectories == replay_trajectories == expected_trajectories):
            raise ValueError("Offline dataset manifest and replay size do not match.")
        if int(manifest.get("total_transitions", -1)) != int(
            replay_metadata.get("total_samples", -1)
        ):
            raise ValueError(
                "Offline dataset manifest and replay transition counts do not match."
            )
        if self._replay_sampling_weight_key is not None and (
            manifest.get("bc_sampling", {}).get("weight_key")
            != self._replay_sampling_weight_key
        ):
            raise ValueError(
                "Offline dataset sampling-weight key does not match BC config."
            )
        self._offline_data_format_version = (
            int(data_version) if data_version is not None else None
        )

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        intervene_traj_list = []
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            intervene_trajs = traj.extract_intervene_traj(mode="all")
            if intervene_trajs is not None:
                intervene_traj_list.extend(intervene_trajs)
        if intervene_traj_list:
            self.replay_buffer.add_trajectories(intervene_traj_list)

    def _prepare_sft_batch(self, batch):
        """Prepare model-specific DAgger training inputs."""
        return self.model.prepare_dagger_sft_batch(batch)

    def _reduce_sft_loss(self, loss):
        """Reduce model-specific SFT loss to a scalar."""
        if not isinstance(loss, torch.Tensor):
            loss = torch.as_tensor(loss, device=self.device)

        if (
            get_supported_model(self.cfg.actor.model.model_type)
            == SupportedModel.OPENPI
        ):
            action_chunk = self.model.config.action_chunk
            action_dim = self.model.config.action_env_dim
            loss = loss[:, :action_chunk, :action_dim]

        return loss.mean()

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """Run one supervised forward pass for DAgger."""
        data = self._prepare_sft_batch(batch)
        actor_output = self.model(forward_type=ForwardType.SFT, data=data)
        self._last_sft_counts = {}
        if isinstance(actor_output, tuple):
            actor_loss, sft_counts = actor_output
            self._last_sft_counts = {
                key: value.detach() for key, value in sft_counts.items()
            }
        else:
            actor_loss = actor_output
        self._last_action_losses = None
        if (
            get_supported_model(self.cfg.actor.model.model_type)
            == SupportedModel.CNN_POLICY
            and isinstance(actor_loss, torch.Tensor)
            and actor_loss.ndim >= 2
        ):
            reduce_dims = tuple(range(actor_loss.ndim - 1))
            self._last_action_losses = actor_loss.detach().mean(dim=reduce_dims)
        return self._reduce_sft_loss(actor_loss)

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        """Run one replay-buffer update epoch for DAgger."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        with self.worker_timer("sample"):
            global_batch = self.replay_buffer.sample(
                num_chunks=global_batch_size_per_rank,
                sampling_weight_key=self._replay_sampling_weight_key,
                without_replacement=self._sample_without_replacement,
            )
        sampled_batch_size = int(global_batch["actions"].shape[0])
        if sampled_batch_size != global_batch_size_per_rank:
            raise RuntimeError(
                f"Replay buffer returned {sampled_batch_size} transitions, expected "
                f"{global_batch_size_per_rank}."
            )

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for idx, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                _apply_sft_drq(batch)
            train_micro_batch_list[idx] = batch

        self.optimizer.zero_grad()
        gbs_actor_loss = []
        gbs_action_losses = []
        gbs_sft_counts: dict[str, list[torch.Tensor]] = {}
        for mb_idx, batch in enumerate(train_micro_batch_list):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
            )
            with self.amp_context:
                actor_loss = self.forward_actor(batch["forward_inputs"])
            if self._last_action_losses is not None:
                gbs_action_losses.append(self._last_action_losses)
            for key, value in self._last_sft_counts.items():
                gbs_sft_counts.setdefault(key, []).append(value)
            actor_loss = actor_loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(actor_loss).backward()
            gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)

        actor_grad_norm, lr_list = self.optimizer_step()
        self.lr_scheduler.step()

        metrics = {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": lr_list[0],
            "actor/grad_norm": actor_grad_norm,
        }
        if gbs_action_losses:
            action_names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
            mean_action_losses = torch.stack(gbs_action_losses).mean(dim=0)
            for action_index, action_loss in enumerate(mean_action_losses):
                action_name = (
                    action_names[action_index]
                    if action_index < len(action_names)
                    else str(action_index)
                )
                metrics[f"dagger/action_{action_name}_loss"] = action_loss
        for metric_name in (
            "binary_accuracy",
            "binary_close_recall",
            "binary_open_recall",
            "binary_transition_close_recall",
            "binary_transition_open_recall",
        ):
            correct_key = f"{metric_name}_correct"
            count_key = f"{metric_name}_count"
            if correct_key not in gbs_sft_counts or count_key not in gbs_sft_counts:
                continue
            correct = torch.stack(gbs_sft_counts[correct_key]).sum()
            count = torch.stack(gbs_sft_counts[count_key]).sum()
            if count.item() > 0:
                metrics[f"dagger/{metric_name}"] = correct / count
        return metrics

    def process_train_metrics(self, metrics):
        """Aggregate DAgger training and replay-buffer metrics."""
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                cpu_values = [
                    v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                mean_metric_dict[key] = (
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )

        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    @Worker.timer("run_training")
    def run_training(self):
        """Run DAgger updates with replay-buffer samples."""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return self.process_train_metrics(metrics)

    def compute_advantages_and_returns(self):
        """Skip advantage computation for supervised DAgger updates."""
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        if self._offline_replay_load_path is None:
            buffer_save_path = os.path.join(
                save_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.save_checkpoint(buffer_save_path)
        sampling_state_path = os.path.join(
            save_base_path,
            f"dagger_components/replay_sampling_state_rank_{self._rank}.pt",
        )
        os.makedirs(os.path.dirname(sampling_state_path), exist_ok=True)
        torch.save(
            {
                "sampling_state": self.replay_buffer.sampling_state_dict(),
                "update_step": self.update_step,
                "world_size": self._world_size,
                "offline_load_path": os.path.realpath(self._offline_replay_load_path)
                if self._offline_replay_load_path
                else None,
                "trajectory_ids": self.replay_buffer.trajectory_ids,
                "total_samples": self.replay_buffer.total_samples,
            },
            sampling_state_path,
        )
        if self._rank == 0:
            torch.save(
                {
                    "format_version": 2,
                    "training_stage": "bc",
                    "data_format_version": self._offline_data_format_version,
                    "model_type": self.cfg.actor.model.model_type,
                    "binary_action_indices": tuple(
                        self.cfg.actor.model.get("binary_action_indices", [])
                    ),
                    "binary_action_temperature": float(
                        self.cfg.actor.model.get("binary_action_temperature", 1.0)
                    ),
                    "binary_loss": "bce_with_logits",
                    "continuous_action_distribution": "tanh_normal_v1",
                    "state_mean": tuple(self.cfg.actor.model.get("state_mean", [])),
                    "state_std": tuple(self.cfg.actor.model.get("state_std", [])),
                },
                os.path.join(save_base_path, "bc_policy_state.pt"),
            )

    def load_checkpoint(self, load_base_path):
        self._validate_bc_checkpoint_marker(load_base_path)
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        if self._offline_replay_load_path is None:
            buffer_load_path = os.path.join(
                load_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.load_checkpoint(buffer_load_path)
        sampling_state_path = os.path.join(
            load_base_path,
            f"dagger_components/replay_sampling_state_rank_{self._rank}.pt",
        )
        if os.path.isfile(sampling_state_path):
            replay_state = torch.load(
                sampling_state_path, map_location="cpu", weights_only=True
            )
            if "sampling_state" not in replay_state:
                replay_state = {"sampling_state": replay_state}
            saved_world_size = replay_state.get("world_size", self._world_size)
            saved_ids = replay_state.get(
                "trajectory_ids", self.replay_buffer.trajectory_ids
            )
            saved_total = replay_state.get(
                "total_samples", self.replay_buffer.total_samples
            )
            if (
                saved_world_size != self._world_size
                or tuple(saved_ids) != self.replay_buffer.trajectory_ids
                or saved_total != self.replay_buffer.total_samples
            ):
                raise ValueError(
                    "Cannot resume BC with a different replay dataset partition."
                )
            saved_path = replay_state.get("offline_load_path")
            current_path = (
                os.path.realpath(self._offline_replay_load_path)
                if self._offline_replay_load_path
                else None
            )
            if saved_path and saved_path != current_path:
                raise ValueError("Cannot resume BC from a different offline dataset.")
            self.replay_buffer.load_sampling_state_dict(replay_state["sampling_state"])
            self.update_step = int(replay_state.get("update_step", self.update_step))

    def _validate_bc_checkpoint_marker(self, load_base_path: str) -> None:
        """Reject incompatible binary-action BC resumes before optimizer loading."""
        binary_indices = tuple(self.cfg.actor.model.get("binary_action_indices", []))
        if not binary_indices:
            return
        marker_path = os.path.join(load_base_path, "bc_policy_state.pt")
        if not os.path.isfile(marker_path):
            raise ValueError("Binary-action BC checkpoint has no format marker.")
        marker = torch.load(marker_path, map_location="cpu", weights_only=True)
        expected = {
            "format_version": 2,
            "training_stage": "bc",
            "data_format_version": self._offline_data_format_version,
            "model_type": self.cfg.actor.model.model_type,
            "binary_action_indices": binary_indices,
            "binary_action_temperature": float(
                self.cfg.actor.model.get("binary_action_temperature", 1.0)
            ),
            "binary_loss": "bce_with_logits",
            "continuous_action_distribution": "tanh_normal_v1",
            "state_mean": tuple(self.cfg.actor.model.get("state_mean", [])),
            "state_std": tuple(self.cfg.actor.model.get("state_std", [])),
        }
        actual = {key: marker.get(key) for key in expected}
        actual["binary_action_indices"] = tuple(
            actual.get("binary_action_indices") or ()
        )
        actual["state_mean"] = tuple(actual.get("state_mean") or ())
        actual["state_std"] = tuple(actual.get("state_std") or ())
        if actual != expected:
            raise ValueError("Binary-action BC checkpoint format is incompatible.")
