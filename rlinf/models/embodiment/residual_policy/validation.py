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

import math
from numbers import Real

from omegaconf import DictConfig, ListConfig


def validate_success_only_reward_cfg(env_cfg: DictConfig) -> None:
    """Reject dense, milestone and relative rewards for residual PnP runs."""
    task = env_cfg.init_params.get("task_alignment", {})
    if (
        env_cfg.init_params.id != "PickAndPlaceDigitalTwin-v1"
        or env_cfg.init_params.reward_mode != "dense"
        or env_cfg.reward_mode != "raw"
        or env_cfg.use_rel_reward
        or env_cfg.ignore_terminations
        or task.get("use_dense_reward", False)
        or task.get("reward_scale", 1.0) != 1.0
        or task.get("sparse_success_reward", 1.0) != 1.0
        or any(
            task.get(key) != 0.0
            for key in (
                "sparse_grasp_reward",
                "sparse_lift_reward",
                "sparse_place_reward",
                "sparse_drop_penalty",
            )
        )
    ):
        raise ValueError(
            "Residual PnP requires the stable success-only reward: reward_mode=dense "
            "in init_params, raw wrapper rewards, use_dense_reward=false, zero "
            "stage rewards/drop penalty, success reward=1, and terminations enabled."
        )


def validate_residual_cfg(cfg: DictConfig) -> None:
    """Reject mismatched residual/base/environment contracts before allocation."""
    model = cfg.actor.model
    if not cfg.rollout.get("residual_base_inference", False):
        raise ValueError("ResidualPolicy requires rollout-side base inference.")
    if cfg.get("reward", {}).get("use_reward_model", False):
        raise ValueError("Residual PnP success-only rewards forbid reward models.")
    if (
        cfg.algorithm.loss_type != "decoupled_actor_critic"
        or cfg.algorithm.adv_type != "gae"
        or model.num_action_chunks != 1
        or not model.add_value_head
        or model.add_q_head
    ):
        raise ValueError(
            "ResidualPolicy requires GAE/PPO, a value head and num_action_chunks=1."
        )
    if not cfg.runner.only_eval and not model.enabled:
        raise ValueError("Disable residual only for runner.only_eval=true baselines.")
    if str(model.precision) != "32" or not cfg.actor.fsdp_config.use_orig_params:
        raise ValueError(
            "Residual PPO requires precision='32' and use_orig_params=true."
        )
    if not cfg.rollout.collect_prev_infos:
        raise ValueError("Residual PPO requires saved logprobs and values.")
    if cfg.rollout.get("expert_model"):
        raise ValueError("Residual PPO does not support expert action relabeling.")
    if model.get("encoder_input_size", 128) != 128:
        raise ValueError(
            "Residual CNN encoder_input_size must be 128; environment images stay 224."
        )
    if model.encoder_config.get("dropout", 0.0) != 0.0:
        raise ValueError(
            "Keep CNN dropout disabled for PPO rollout/update consistency."
        )
    active = list(model.residual_action_indices)
    scale = list(model.residual_action_scale)
    gripper_mode = model.get("gripper_mode", "cnn")
    if gripper_mode not in ("residual", "cnn", "none"):
        raise ValueError("gripper_mode must be residual, cnn or none.")
    if gripper_mode == "none":
        if (
            model.action_dim != 6
            or active != list(range(6))
            or model.get("gripper_architecture", "shared") != "shared"
            or not model.get("gripper_share_features", True)
            or model.get("gripper_checkpoint")
            or model.get("shared_feature_checkpoint")
            or model.get("gripper")
            or model.get("binary_action_indices")
            or not model.get("independent_std", True)
        ):
            raise ValueError(
                "No-gripper residual requires six continuous actions and no gripper BC."
            )
        if cfg.algorithm.logprob_type != "chunk_level":
            raise ValueError("No-gripper residual requires joint chunk_level logprobs.")
        if cfg.rollout.get("enable_cuda_graph", False) or cfg.rollout.get(
            "enable_torch_compile", False
        ):
            raise ValueError("Keep residual rollout compile and CUDA graphs disabled.")
    if gripper_mode == "cnn":
        architecture = model.get("gripper_architecture", "shared")
        if architecture not in ("shared", "independent"):
            raise ValueError("gripper_architecture must be shared or independent.")
        share_features = model.get("gripper_share_features", True)
        if not isinstance(share_features, bool):
            raise ValueError("gripper_share_features must be a boolean.")
        if architecture != "shared" and not share_features:
            raise ValueError(
                "gripper_share_features=false requires gripper_architecture=shared."
            )
        if architecture == "shared":
            if model.get("gripper"):
                raise ValueError(
                    "Shared gripper does not use the independent gripper config. "
                    "Remove it or select gripper_architecture=independent."
                )
            if not model.get("independent_std", True):
                raise ValueError("Shared gripper requires independent_std=true.")
            if model.get("gripper_checkpoint") and not model.get(
                "shared_feature_checkpoint"
            ):
                raise ValueError(
                    "Shared gripper BC requires shared_feature_checkpoint."
                )
        if model.action_dim != 7 or active != list(range(6)):
            raise ValueError(
                "CNN gripper requires exactly six arm residual dimensions."
            )
        if model.get("binary_action_indices"):
            raise ValueError(
                "CNN gripper uses its own Bernoulli head, not CNNPolicy binary indices."
            )
        if cfg.algorithm.logprob_type != "chunk_level":
            raise ValueError("Split gripper PPO requires joint chunk_level logprobs.")
        if cfg.rollout.get("enable_cuda_graph", False) or cfg.rollout.get(
            "enable_torch_compile", False
        ):
            raise ValueError(
                "Split gripper currently requires rollout compile and CUDA graphs disabled."
            )
        if not (
            model.get("gripper_checkpoint")
            or cfg.runner.get("resume_dir")
            or cfg.runner.get("ckpt_path")
        ):
            raise ValueError(
                "Provide gripper_checkpoint from BC or a full joint-policy checkpoint."
            )
    if (
        not active
        or len(set(active)) != len(active)
        or any(i < 0 or i >= model.action_dim for i in active)
    ):
        raise ValueError("Invalid residual_action_indices.")
    if len(scale) != model.action_dim or any(
        not math.isfinite(s) or s < 0 for s in scale
    ):
        raise ValueError(
            "Residual scales must be finite nonnegative action_dim entries."
        )
    if any((scale[i] > 0) != (i in active) for i in range(model.action_dim)):
        raise ValueError(
            "Positive residual scales must match residual_action_indices exactly."
        )
    initial_logstd = model.initial_logstd
    if isinstance(initial_logstd, (list, tuple, ListConfig)):
        if len(initial_logstd) != model.action_dim:
            raise ValueError("initial_logstd must contain action_dim entries.")
        initial_values = list(initial_logstd)
    else:
        initial_values = [initial_logstd]
    bounds = model.logstd_range
    if (
        len(bounds) != 2
        or any(not isinstance(v, Real) or not math.isfinite(v) for v in bounds)
        or bounds[0] > bounds[1]
        or any(
            not isinstance(v, Real)
            or not math.isfinite(v)
            or not bounds[0] <= v <= bounds[1]
            for v in initial_values
        )
    ):
        raise ValueError("initial_logstd must lie inside finite logstd_range bounds.")

    _validate_rollout_contract(cfg)


def _validate_rollout_contract(cfg: DictConfig) -> None:
    model = cfg.actor.model
    peg = model.get("gripper_mode") == "none"
    if model.get("gripper_mode") not in ("cnn", "none") or not model.use_state:
        raise ValueError(
            "Rollout-side residual requires CNN gripper and proprioception."
        )
    if (model.state_dim, model.image_num) != ((6, 1) if peg else (14, 2)):
        raise ValueError(
            "Use wrist/6-D state for Peg, or two views/14-D state for PnP."
        )
    if (
        model.get("gripper_architecture", "shared") == "independent"
        and model.get("gripper", {}).get("state_dim") != 14
    ):
        raise ValueError("CNN gripper must also receive the 14-D proprioceptive state.")
    if model.get("state_mean") or model.get("state_std"):
        raise ValueError(
            "Do not apply 14-D CNN normalization to the augmented residual state."
        )
    co_training = cfg.algorithm.get("sim_real_rl_co_training", False)
    simulation_only = not co_training and cfg.env.train.env_type == "maniskill"
    real_only = not co_training and cfg.env.train.env_type == "realworld"
    standalone_eval = (
        cfg.runner.only_eval
        and real_only
        and cfg.get("evaluation", {}).get("num_episodes", 0) > 0
    )
    if not (co_training or simulation_only or real_only):
        raise ValueError("Residual requires ManiSkill or realworld environments.")
    expected_nodes = 3 if co_training else 1 if simulation_only else 2
    if cfg.cluster.num_nodes != expected_nodes or cfg.rollout.pipeline_stage_num != 1:
        raise ValueError(
            f"Rollout-side residual requires {expected_nodes} nodes and one stage."
        )
    if cfg.algorithm.get("bootstrap_type") != "none":
        raise ValueError(
            "Rollout-side residual currently requires bootstrap_type=none."
        )
    if cfg.runner.val_check_interval > 0:
        raise ValueError(
            "Residual training must run without evaluation; periodic eval is disabled."
        )
    if cfg.runner.only_eval and not standalone_eval:
        raise ValueError(
            "Use standalone real-world evaluation; other residual modes run "
            "without evaluation."
        )
    if cfg.rollout.get("collect_transitions", False) or cfg.rollout.enable_offload:
        raise ValueError(
            "Rollout-side residual requires PPO forward_inputs and no offload."
        )
    if cfg.actor.get("sync_weight_no_wait", False):
        raise ValueError("Use the existing PPO weight synchronization barrier.")
    if (
        co_training
        and cfg.algorithm.get("co_training_rollout_routing_mode") != "paired"
    ):
        raise ValueError("Rollout-side residual co-training requires paired routing.")
    for domain in (cfg.env.train, cfg.env.eval):
        if simulation_only:
            _validate_simulation_contract(domain, peg=peg)
            continue
        real = domain.co_training_env_cfg if co_training else domain
        if (
            real.env_type != "realworld"
            or real.total_num_envs != 1
            or real.num_workers != 1
        ):
            raise ValueError("Use one realworld environment and one real worker.")
        expected_state_keys = (
            ["ee_target_delta"]
            if peg
            else [
                "arm_joint_position",
                "tcp_pose",
                "gripper_open_state",
            ]
        )
        if list(real.get("state_keys", [])) != expected_state_keys:
            raise ValueError(
                "Real state_keys must match the joint/pose/gripper BC order."
            )
        if real.get("use_spacemouse", False) or real.get("use_gello", False):
            raise ValueError(
                "Residual PPO does not support intervention action relabeling."
            )
        if real.get("keyboard_reward_wrapper") != (None if peg else "pnp_human"):
            raise ValueError(
                "Use automatic Peg success or the PnP human reward wrapper."
            )
        if peg and (
            real.init_params.id != "FrankaCoTrainingPegInsertionEnv-v1"
            or real.get("main_image_key") != "wrist_1"
            or real.override_cfg.peg_config.get("dense_reward_scale", 0.1) != 0.0
        ):
            raise ValueError("Peg requires its wrist environment and sparse rewards.")
        if not real.include_states_in_obs or real.ignore_terminations:
            raise ValueError("Real residual requires states and terminations.")
        if real.auto_reset != (not standalone_eval):
            raise ValueError(
                "Real residual requires auto_reset=false for standalone eval "
                "and auto_reset=true for training."
            )
        if domain.get("residual") or real.get("residual"):
            raise ValueError(
                "Do not enable an env-side residual wrapper with rollout composition."
            )
        if co_training:
            _validate_simulation_contract(domain, peg=peg)
    base = cfg.base_model
    if (
        base.model_type != "openpi"
        or base.action_dim != (6 if peg else 7)
        or base.openpi.get("use_dsrl", False)
    ):
        raise ValueError(
            "Use a frozen OpenPI SFT base matching the task action dimensions."
        )
    if peg and (
        base.openpi.config_name != "pi05_peg_wrist_state"
        or base.openpi.action_env_dim != 6
        or base.openpi.num_images_in_input != 1
        or not base.openpi.get("discrete_state_input", False)
    ):
        raise ValueError(
            "Peg SFT requires its wrist/state adapter and six-axis outputs."
        )
    if model.base_horizon > min(
        base.num_action_chunks, base.openpi.action_chunk, base.openpi.action_horizon
    ):
        raise ValueError("base_horizon exceeds the base model horizon.")


def _validate_simulation_contract(domain: DictConfig, *, peg: bool = False) -> None:
    if domain.env_type != "maniskill" or domain.num_workers != 1:
        raise ValueError("Use one ManiSkill worker paired with one simulation rollout.")
    if domain.get("residual"):
        raise ValueError(
            "Do not enable an env-side residual wrapper with rollout composition."
        )
    if (
        not domain.include_states_in_obs
        or not domain.auto_reset
        or domain.get("enable_offload", False)
    ):
        raise ValueError("Simulation requires states, auto reset, and no offload.")
    if domain.init_params.control_mode != "pd_ee_body_target_delta_pose_real":
        raise ValueError("Simulation must use the normalized PnP delta controller.")
    if peg:
        if (
            domain.init_params.id != "PegInsertionDigitalTwin-v1"
            or domain.init_params.reward_mode != "dense"
            or domain.reward_mode != "raw"
            or domain.use_rel_reward
            or domain.ignore_terminations
            or domain.init_params.get("peg_config", {}).get("dense_reward_scale", 0.1)
            != 0.0
        ):
            raise ValueError(
                "Peg residual requires its native success-only reward and terminations."
            )
        return
    validate_success_only_reward_cfg(domain)
    control = domain.init_params.controller_alignment
    if not control.binary_gripper_action or control.use_zero_one_gripper_action:
        raise ValueError("Simulation requires binary +/-1 gripper commands.")
    if control.open_command != 1.0 or control.close_command != -1.0:
        raise ValueError("Gripper convention must be -1 close, +1 open.")
