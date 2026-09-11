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
    if cfg.rollout.pipeline_stage_num != 1 or cfg.cluster.num_nodes != 1:
        raise ValueError(
            "The first residual implementation supports one simulation node/stage."
        )
    if not cfg.rollout.collect_prev_infos or cfg.algorithm.get(
        "sim_real_rl_co_training", False
    ):
        raise ValueError(
            "Residual PPO requires old logprobs/values and simulation-only training."
        )
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
    gripper_mode = model.get("gripper_mode", "residual")
    if gripper_mode not in ("residual", "cnn"):
        raise ValueError("gripper_mode must be residual or cnn.")
    if gripper_mode == "cnn":
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
    for env_cfg in (cfg.env.train, cfg.env.eval):
        validate_success_only_reward_cfg(env_cfg)
        residual = env_cfg.get("residual")
        if env_cfg.env_type != "maniskill" or residual is None:
            raise ValueError("ResidualPolicy needs a ManiSkill residual environment.")
        if residual.get("gripper_mode", "residual") != gripper_mode:
            raise ValueError("Environment and policy gripper_mode must match.")
        if gripper_mode == "cnn":
            controller = env_cfg.init_params.get("controller_alignment", {})
            if (
                not controller.get("binary_gripper_action", False)
                or controller.get("use_zero_one_gripper_action", False)
                or controller.get("open_command", 1.0) != 1.0
                or controller.get("close_command", -1.0) != -1.0
                or not -1.0 < controller.get("binary_gripper_threshold", 0.5) < 1.0
            ):
                raise ValueError(
                    "CNN gripper requires binary controller commands: -1 close, +1 open."
                )
        if not env_cfg.include_states_in_obs or env_cfg.get("enable_offload", False):
            raise ValueError(
                "Keep original base states enabled and environment offload disabled."
            )
        if env_cfg.init_params.control_mode != "pd_ee_body_target_delta_pose_real":
            raise ValueError(
                "This residual example requires the normalized PnP delta controller."
            )
        for key in ("action_dim", "base_horizon", "use_state"):
            if residual[key] != model[key]:
                raise ValueError(
                    f"Environment residual.{key} must match actor.model.{key}."
                )
        if list(residual.action_scale) != scale:
            raise ValueError(
                "Environment residual scales must match the model configuration."
            )
        base = residual.base_model
        if base.model_type != "openpi" or base.action_dim != model.action_dim:
            raise ValueError("Base must be OpenPI with matching action_dim.")
        if base.openpi.get("use_dsrl", False):
            raise ValueError("The frozen SFT base must not enable DSRL.")
        if model.base_horizon > min(
            base.num_action_chunks, base.openpi.action_chunk, base.openpi.action_horizon
        ):
            raise ValueError("base_horizon exceeds the base model action horizon.")
