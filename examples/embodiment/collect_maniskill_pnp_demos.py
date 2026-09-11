from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gymnasium as gym
import hydra
import imageio.v2 as imageio
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import rlinf.envs.maniskill.tasks.digital_twin  # noqa: F401
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.maniskill.tasks.digital_twin.scripted_pick_and_place_expert import (
    ScriptedPickAndPlaceExpert,
    ScriptedPickAndPlaceExpertConfig,
    validate_persistent_gripper_commands,
)


def build_action_sampling_metadata(
    actions: torch.Tensor,
    transition_weight: float,
    gripper_action_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build generic sampling weights around binary-action transitions.

    Args:
        actions: Action sequence with time as its first dimension.
        transition_weight: Relative weight assigned to transition frames.
        gripper_action_index: Binary-action channel in the last dimension.

    Returns:
        Per-transition weights and a boolean transition mask.
    """
    if transition_weight < 1.0:
        raise ValueError("transition_weight must be at least 1.0.")
    gripper_actions = actions[..., gripper_action_index]
    transitions = torch.zeros_like(gripper_actions, dtype=torch.bool)
    transitions[1:] = gripper_actions[1:] != gripper_actions[:-1]
    weights = torch.ones_like(gripper_actions, dtype=torch.float32)
    weights[transitions] = transition_weight
    return weights.unsqueeze(-1), transitions.unsqueeze(-1)


class DemoDiversityTracker:
    """Reject quantized duplicates and report transition-level coverage."""

    def __init__(
        self,
        initial_state_resolution: float,
        action_resolution: float,
        reject_near_duplicates: bool,
    ) -> None:
        if initial_state_resolution <= 0 or action_resolution <= 0:
            raise ValueError("Diversity resolutions must be positive.")
        self.initial_state_resolution = initial_state_resolution
        self.action_resolution = action_resolution
        self.reject_near_duplicates = reject_near_duplicates
        self._initial_fingerprints: set[tuple[int, ...]] = set()
        self._trajectory_fingerprints: set[str] = set()
        self._accepted_initial_states: list[torch.Tensor] = []
        self._state_action_fingerprints: set[tuple[int, ...]] = set()
        self.accepted_transition_count = 0
        self.candidates = 0
        self.unique_candidates = 0
        self.duplicate_initial_state_candidates = 0
        self.duplicate_trajectory_candidates = 0

    @staticmethod
    def _tensor_digest(tensor: torch.Tensor) -> str:
        contiguous = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

    def consider(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
        task_states: torch.Tensor | None = None,
    ) -> tuple[bool, str | None]:
        """Return whether a successful candidate adds effective diversity.

        Args:
            initial_state: Compact task-relevant initial-state descriptor.
            actions: Complete successful action trajectory.
            task_states: Optional per-step task pose used for coverage reporting.

        Returns:
            Acceptance flag and duplicate category, if any.
        """
        self.candidates += 1
        initial_fingerprint = tuple(
            torch.round(initial_state.detach().cpu() / self.initial_state_resolution)
            .to(torch.int64)
            .tolist()
        )
        quantized_actions = torch.round(
            actions.detach().cpu() / self.action_resolution
        ).to(torch.int32)
        trajectory_fingerprint = self._tensor_digest(quantized_actions)

        reason = None
        is_near_initial_state = any(
            torch.all(
                torch.abs(initial_state.detach().cpu() - accepted_state)
                <= self.initial_state_resolution
            ).item()
            for accepted_state in self._accepted_initial_states
        )
        if is_near_initial_state:
            reason = "initial_state"
            self.duplicate_initial_state_candidates += 1
        elif trajectory_fingerprint in self._trajectory_fingerprints:
            reason = "trajectory"
            self.duplicate_trajectory_candidates += 1
        else:
            self.unique_candidates += 1

        if reason is not None and self.reject_near_duplicates:
            return False, reason

        self._initial_fingerprints.add(initial_fingerprint)
        self._trajectory_fingerprints.add(trajectory_fingerprint)
        self._accepted_initial_states.append(initial_state.detach().cpu().float())
        if task_states is not None:
            flat_states = task_states.detach().cpu().reshape(actions.shape[0], -1)
            flat_actions = actions.detach().cpu().reshape(actions.shape[0], -1)
            if flat_states.shape[0] != flat_actions.shape[0]:
                raise ValueError("Task-state and action trajectory lengths must match.")
            quantized_state_actions = torch.cat(
                [
                    torch.round(flat_states / self.initial_state_resolution),
                    torch.round(flat_actions / self.action_resolution),
                ],
                dim=-1,
            ).to(torch.int32)
            self._state_action_fingerprints.update(
                tuple(row.tolist()) for row in quantized_state_actions
            )
            self.accepted_transition_count += flat_actions.shape[0]
        return True, reason

    def summary(self) -> dict[str, object]:
        """Return compact diversity and duplicate statistics."""
        accepted = len(self._accepted_initial_states)
        if accepted:
            initial_states = torch.stack(self._accepted_initial_states)
            state_stats = {
                "min": initial_states.amin(dim=0).tolist(),
                "max": initial_states.amax(dim=0).tolist(),
                "std": initial_states.std(dim=0, unbiased=False).tolist(),
            }
        else:
            state_stats = {"min": [], "max": [], "std": []}
        return {
            "candidate_episodes": self.candidates,
            "accepted_episodes": accepted,
            "accepted_candidate_ratio": accepted / max(self.candidates, 1),
            "unique_candidate_ratio": self.unique_candidates / max(self.candidates, 1),
            "duplicate_initial_state_candidates": (
                self.duplicate_initial_state_candidates
            ),
            "duplicate_action_trajectory_candidates": (
                self.duplicate_trajectory_candidates
            ),
            "rejected_quantized_duplicate_episodes": (
                self.duplicate_initial_state_candidates
                + self.duplicate_trajectory_candidates
                if self.reject_near_duplicates
                else 0
            ),
            "unique_initial_fingerprints": len(self._initial_fingerprints),
            "unique_trajectory_fingerprints": len(self._trajectory_fingerprints),
            "unique_quantized_state_action_transitions": len(
                self._state_action_fingerprints
            ),
            "quantized_state_action_ratio": len(self._state_action_fingerprints)
            / max(self.accepted_transition_count, 1),
            "initial_state_descriptor": "cube_pose,tcp_pose",
            "initial_state_resolution": self.initial_state_resolution,
            "action_resolution": self.action_resolution,
            "reject_near_duplicates": self.reject_near_duplicates,
            "accepted_initial_state_stats": state_stats,
        }


def _save_rgb_video(path: Path, frames: torch.Tensor, fps: int) -> None:
    """Save a [time, height, width, rgb] uint8 tensor as an MP4 file."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected [T, H, W, 3] RGB frames, got {frames.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    try:
        for frame in frames:
            writer.append_data(frame.numpy())
    finally:
        writer.close()


def _add_target_centered_safety_limits(env_kwargs: dict) -> None:
    controller = env_kwargs.get("controller_alignment", {})
    if "ee_pose_limit_min" in controller and "ee_pose_limit_max" in controller:
        return

    target = controller.get("target_ee_pose")
    if target is None:
        return
    clip_x = float(controller.get("clip_x_range", 0.0))
    clip_y = float(controller.get("clip_y_range", 0.0))
    clip_z_high = float(controller.get("clip_z_range_high", 0.0))
    clip_z_low = float(controller.get("clip_z_range_low", 0.0))
    clip_rz = float(controller.get("clip_rz_range", 0.0))
    clip_rp = float(controller.get("clip_rp_range", 0.0))
    controller["ee_pose_limit_min"] = [
        target[0] - clip_x,
        target[1] - clip_y,
        target[2] - clip_z_low,
        target[3] - clip_rp,
        target[4] - clip_rp,
        target[5] - clip_rz,
    ]
    controller["ee_pose_limit_max"] = [
        target[0] + clip_x,
        target[1] + clip_y,
        target[2] + clip_z_high,
        target[3] + clip_rp,
        target[4] + clip_rp,
        target[5] + clip_rz,
    ]


def _build_trajectory(
    env_index: int,
    success_step: int,
    max_episode_steps: int,
    main_images: list[torch.Tensor],
    states: list[torch.Tensor],
    task_states: list[torch.Tensor],
    actions: list[torch.Tensor],
    rewards: list[torch.Tensor],
    terminations: list[torch.Tensor],
    truncations: list[torch.Tensor],
    transition_sampling_weight: float,
) -> Trajectory:
    end = success_step + 1

    def stack_env(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values[:end], dim=0)[:, env_index : env_index + 1]

    action_tensor = stack_env(actions)
    reward_tensor = stack_env(rewards)
    termination_tensor = stack_env(terminations)
    truncation_tensor = stack_env(truncations)
    done_tensor = termination_tensor | truncation_tensor
    termination_tensor[-1] = True
    done_tensor[-1] = True
    sample_weight, action_transition = build_action_sampling_metadata(
        action_tensor, transition_sampling_weight
    )

    forward_inputs = {
        "main_images": stack_env(main_images),
        "states": stack_env(states),
        "task_states": stack_env(task_states),
        "action": action_tensor.clone(),
        "sample_weight": sample_weight,
        "action_transition": action_transition,
    }
    return Trajectory(
        max_episode_length=max_episode_steps,
        model_weights_id="scripted_expert",
        actions=action_tensor,
        intervene_flags=torch.ones_like(action_tensor, dtype=torch.bool),
        rewards=reward_tensor,
        terminations=termination_tensor,
        truncations=truncation_tensor,
        dones=done_tensor,
        forward_inputs=forward_inputs,
    )


def validate_demonstration_trajectory(trajectory: Trajectory) -> None:
    """Validate one successful demonstration before it reaches BC storage."""
    inputs = trajectory.forward_inputs
    required = (
        "main_images",
        "states",
        "task_states",
        "action",
        "sample_weight",
        "action_transition",
    )
    missing = [name for name in required if name not in inputs]
    if missing:
        raise ValueError(f"Demonstration is missing fields: {missing}.")

    actions = inputs["action"]
    trajectory_length = actions.shape[0]
    for name in required:
        if inputs[name].shape[0] != trajectory_length:
            raise ValueError(f"Demonstration field {name!r} has a mismatched length.")
    if trajectory.actions.shape != actions.shape or not torch.equal(
        trajectory.actions, actions
    ):
        raise ValueError("Stored policy targets do not match executed actions.")
    if inputs["main_images"].dtype != torch.uint8:
        raise ValueError("Demonstration RGB images must be uint8.")
    if inputs["main_images"].shape[-1] != 3:
        raise ValueError("Demonstration RGB images must have three channels.")
    if inputs["states"].shape[-1] != 14:
        raise ValueError("Demonstration policy states must have 14 features.")
    if inputs["task_states"].shape[-1] != 14:
        raise ValueError("Demonstration audit states must have 14 features.")
    if actions.shape[-1] != 7:
        raise ValueError("Demonstration actions must have seven channels.")
    transition_flags = inputs["action_transition"]
    sample_weights = inputs["sample_weight"]
    expected_metadata_shape = (*actions.shape[:-1], 1)
    if transition_flags.dtype != torch.bool:
        raise ValueError("Demonstration action-transition flags must be boolean.")
    if (
        transition_flags.shape != expected_metadata_shape
        or sample_weights.shape != expected_metadata_shape
    ):
        raise ValueError("Demonstration sampling metadata has an invalid shape.")
    expected_transitions = torch.zeros_like(transition_flags)
    expected_transitions[1:] = (
        (actions[1:, ..., 6] > 0) != (actions[:-1, ..., 6] > 0)
    ).unsqueeze(-1)
    if not torch.equal(transition_flags, expected_transitions):
        raise ValueError("Demonstration action-transition metadata is inconsistent.")
    for name in ("states", "task_states", "action", "sample_weight"):
        if not torch.isfinite(inputs[name]).all():
            raise ValueError(
                f"Demonstration field {name!r} contains non-finite values."
            )
    if torch.any(torch.abs(actions) > 1.0 + 1e-6):
        raise ValueError("Demonstration actions must lie in [-1, 1].")
    if torch.any(inputs["sample_weight"] < 1.0):
        raise ValueError("Demonstration sampling weights must be at least one.")
    if not torch.all(sample_weights[~transition_flags] == 1.0):
        raise ValueError("Non-transition samples must retain unit sampling weight.")
    if not bool(trajectory.dones[-1].all()):
        raise ValueError("A successful demonstration must terminate on its final step.")


def collect_demos(cfg: DictConfig) -> None:
    output_dir = Path(cfg.collector.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Demo output directory is not empty: {output_dir}. Use a new directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = OmegaConf.to_container(cfg.env.train.init_params, resolve=True)
    env_kwargs["num_envs"] = int(cfg.collector.num_envs)
    _add_target_centered_safety_limits(env_kwargs)

    expert_kwargs = OmegaConf.to_container(cfg.collector.expert, resolve=True)
    expert_kwargs["position_action_scale"] = float(
        env_kwargs["controller_alignment"]["action_scale"][0]
    )
    expert_config = ScriptedPickAndPlaceExpertConfig(**expert_kwargs)

    replay_buffer = TrajectoryReplayBuffer(
        seed=int(cfg.collector.seed),
        enable_cache=False,
        cache_size=0,
        sample_window_size=int(cfg.collector.num_episodes),
        auto_save=True,
        auto_save_path=str(output_dir),
        trajectory_format="pt",
    )
    env = gym.make(**env_kwargs)
    task = env.unwrapped
    expert = ScriptedPickAndPlaceExpert(
        num_envs=int(cfg.collector.num_envs),
        device=task.device,
        config=expert_config,
    )

    target_episodes = int(cfg.collector.num_episodes)
    max_episode_steps = int(cfg.env.train.max_episode_steps)
    max_batches = int(cfg.collector.max_attempt_batches)
    save_video_count = min(int(cfg.collector.save_video_count), target_episodes)
    video_fps = int(cfg.collector.video_fps)
    video_dir = Path(cfg.collector.video_output_dir).expanduser().resolve()
    saved_video_paths: list[str] = []
    gripper_stats: list[dict[str, int]] = []
    diversity_cfg = cfg.collector.diversity
    diversity_tracker = DemoDiversityTracker(
        initial_state_resolution=float(diversity_cfg.initial_state_resolution),
        action_resolution=float(diversity_cfg.action_resolution),
        reject_near_duplicates=bool(diversity_cfg.reject_near_duplicates),
    )
    transition_sampling_weight = float(cfg.collector.transition_sampling_weight)
    rejected_invalid_episodes = 0
    collected = 0
    attempted = 0
    successful_attempts = 0
    progress = tqdm(total=target_episodes, desc="Collecting scripted PnP demos")

    try:
        for batch_index in range(max_batches):
            if collected >= target_episodes:
                break
            _, infos = env.reset(seed=int(cfg.collector.seed) + batch_index)
            extracted_obs = infos["extracted_obs"]
            initial_eval = task.evaluate()
            expert.reset(task.cube.pose.p, initial_eval["goal_position"])
            initial_state_descriptors = (
                torch.cat(
                    [task.cube.pose.raw_pose, task.agent.tcp.pose.raw_pose], dim=-1
                )
                .detach()
                .cpu()
            )

            first_success_step = torch.full(
                (expert.num_envs,), -1, dtype=torch.long, device=task.device
            )
            batch_main_images: list[torch.Tensor] = []
            batch_states: list[torch.Tensor] = []
            batch_task_states: list[torch.Tensor] = []
            batch_actions: list[torch.Tensor] = []
            batch_rewards: list[torch.Tensor] = []
            batch_terminations: list[torch.Tensor] = []
            batch_truncations: list[torch.Tensor] = []

            for step in range(max_episode_steps):
                eval_info = task.evaluate()
                actions = expert.act(
                    tcp_positions=task.agent.tcp.pose.p,
                    cube_positions=task.cube.pose.p,
                    goal_positions=eval_info["goal_position"],
                    is_grasped=eval_info["is_grasped"],
                    is_lifted=eval_info["is_obj_lifted"],
                    is_placed=eval_info["is_obj_placed"],
                    success=eval_info["success"],
                )
                already_successful = first_success_step >= 0
                actions[already_successful, :6] = 0.0
                actions[already_successful, 6] = 1.0

                batch_main_images.append(
                    extracted_obs["main_images"].detach().cpu().contiguous()
                )
                batch_states.append(extracted_obs["states"].detach().cpu().contiguous())
                batch_task_states.append(
                    torch.cat(
                        [task.cube.pose.raw_pose, task.agent.tcp.pose.raw_pose], dim=-1
                    )
                    .detach()
                    .cpu()
                    .contiguous()
                )
                batch_actions.append(actions.detach().cpu().contiguous())

                _, reward, terminations, truncations, infos = env.step(actions)
                extracted_obs = infos["extracted_obs"]
                success = infos["success"].to(dtype=torch.bool)
                new_success = success & (first_success_step < 0)
                first_success_step[new_success] = step

                batch_rewards.append(
                    reward.detach().to(torch.float32).cpu().unsqueeze(1).contiguous()
                )
                batch_terminations.append(
                    terminations.detach().to(torch.bool).cpu().unsqueeze(1).contiguous()
                )
                batch_truncations.append(
                    truncations.detach().to(torch.bool).cpu().unsqueeze(1).contiguous()
                )
                if torch.all(first_success_step >= 0):
                    break

            trajectories = []
            for env_index, success_step in enumerate(first_success_step.tolist()):
                if success_step < 0 or collected + len(trajectories) >= target_episodes:
                    continue
                trajectory = _build_trajectory(
                    env_index=env_index,
                    success_step=success_step,
                    max_episode_steps=max_episode_steps,
                    main_images=batch_main_images,
                    states=batch_states,
                    task_states=batch_task_states,
                    actions=batch_actions,
                    rewards=batch_rewards,
                    terminations=batch_terminations,
                    truncations=batch_truncations,
                    transition_sampling_weight=transition_sampling_weight,
                )
                try:
                    validate_demonstration_trajectory(trajectory)
                    trajectory_gripper_stats = validate_persistent_gripper_commands(
                        trajectory.forward_inputs["action"]
                    )
                except ValueError:
                    rejected_invalid_episodes += 1
                    continue
                is_diverse, _ = diversity_tracker.consider(
                    initial_state_descriptors[env_index],
                    trajectory.actions,
                    trajectory.forward_inputs["task_states"],
                )
                if not is_diverse:
                    continue
                if len(saved_video_paths) < save_video_count:
                    video_path = video_dir / (
                        f"pnp_seed_{int(cfg.collector.seed)}_success_"
                        f"{collected + len(trajectories):04d}.mp4"
                    )
                    _save_rgb_video(
                        video_path,
                        trajectory.forward_inputs["main_images"][:, 0],
                        video_fps,
                    )
                    saved_video_paths.append(str(video_path))
                trajectories.append(trajectory)
                gripper_stats.append(trajectory_gripper_stats)
            replay_buffer.add_trajectories(trajectories)
            collected += len(trajectories)
            attempted += expert.num_envs
            successful_attempts += int((first_success_step >= 0).sum().item())
            progress.update(len(trajectories))
            progress.set_postfix(
                expert_success_rate=f"{successful_attempts / attempted:.3f}"
            )
    finally:
        progress.close()
        env.close()
        replay_buffer.close()

    total_transitions = sum(item["episode_steps"] for item in gripper_stats)
    total_open_steps = sum(item["open_steps"] for item in gripper_stats)
    total_close_steps = sum(item["close_steps"] for item in gripper_stats)
    total_transition_steps = 2 * len(gripper_stats)
    total_sampling_weight = (
        total_transitions + (transition_sampling_weight - 1.0) * total_transition_steps
    )
    diversity_summary = diversity_tracker.summary()
    summary = {
        "format_version": 2,
        "collected_episodes": collected,
        "attempted_episodes": attempted,
        "successful_attempts": successful_attempts,
        "success_rate": successful_attempts / max(attempted, 1),
        "rejected_invalid_episodes": rejected_invalid_episodes,
        "diversity": diversity_summary,
        "total_transitions": total_transitions,
        "gripper_commands": {
            "encoding": {"open": 1.0, "close": -1.0},
            "pattern": "open-close-open",
            "open_steps": total_open_steps,
            "close_steps": total_close_steps,
            "close_fraction": total_close_steps / max(total_transitions, 1),
            "min_close_run_steps": min(
                (item["close_run_steps"] for item in gripper_stats), default=0
            ),
            "transition_sampling_weight": transition_sampling_weight,
        },
        "bc_sampling": {
            "weight_key": "forward_inputs.sample_weight",
            "action_transition_steps": total_transition_steps,
            "uniform_transition_fraction": total_transition_steps
            / max(total_transitions, 1),
            "weighted_transition_fraction": (
                transition_sampling_weight
                * total_transition_steps
                / max(total_sampling_weight, 1.0)
            ),
        },
        "seed": int(cfg.collector.seed),
        "num_envs": int(cfg.collector.num_envs),
        "saved_success_videos": saved_video_paths,
        "environment_randomization": OmegaConf.to_container(
            cfg.env.train.init_params, resolve=True
        ),
        "expert": OmegaConf.to_container(cfg.collector.expert, resolve=True),
    }
    with (output_dir / "collection_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)

    min_unique_ratio = float(diversity_cfg.min_unique_candidate_ratio)
    if diversity_summary["unique_candidate_ratio"] < min_unique_ratio:
        raise RuntimeError(
            "Successful demonstration candidates are insufficiently diverse: "
            f"unique ratio {diversity_summary['unique_candidate_ratio']:.3f} "
            f"< {min_unique_ratio:.3f}. See "
            f"{output_dir / 'collection_summary.json'}."
        )
    if collected < target_episodes:
        raise RuntimeError(
            f"Collected only {collected}/{target_episodes} successful demos after "
            f"{attempted} attempts. See {output_dir / 'collection_summary.json'}."
        )


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_pick_and_place_collect_demos",
)
def main(cfg: DictConfig) -> None:
    collect_demos(cfg)


if __name__ == "__main__":
    main()
