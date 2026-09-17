# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Standalone real-world evaluation: one policy, one robot, exact episode count."""

import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.realworld_eval import append_record, evaluation_checkpoint, summarize
from rlinf.workers.env.realworld_eval_worker import RealworldEvalEnvWorker
from rlinf.workers.rollout.hf.realworld_eval_worker import RealworldEvalRolloutWorker


def validate_eval(cfg):
    """Validate the evaluation contract without touching Ray or hardware."""
    assert cfg.runner.only_eval and not cfg.algorithm.sim_real_rl_co_training
    assert cfg.env.eval.env_type == "realworld"
    assert cfg.env.eval.total_num_envs == 1 and cfg.rollout.pipeline_stage_num == 1
    assert cfg.actor.model.num_action_chunks == 1, (
        "Real evaluation supports one executed action per prediction"
    )
    assert not cfg.env.eval.auto_reset and not cfg.env.eval.ignore_terminations
    assert cfg.evaluation.num_episodes > 0 and cfg.env.eval.max_episode_steps > 0
    assert cfg.evaluation.policy_mode in ("deterministic", "sample")
    assert not cfg.actor.get("initial_checkpoint_exclude_keys", [])
    assert not cfg.rollout.get("expert_model"), (
        "Evaluation cannot use a separate expert"
    )
    assert not cfg.env.eval.get("use_spacemouse", False)
    assert not cfg.env.eval.get("use_gello", False)
    cfg.runner.ckpt_path = evaluation_checkpoint(cfg)
    if cfg.actor.model.model_type == "residual_policy":
        from rlinf.models.embodiment.residual_policy.validation import (
            validate_residual_cfg,
        )

        validate_residual_cfg(cfg)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="real_eval_peg_insertion_cnn_wrist_state",
)
def main(cfg):
    source = hydra.compose(config_name=cfg.evaluation.training_config)
    cfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(source)), cfg)
    validate_eval(cfg)
    root = Path(cfg.runner.logger.log_path)
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "config.yaml", resolve=True)
    if cfg.evaluation.check_only:
        print(
            "Configuration/checkpoint path validated; no Ray workers or robot initialized."
        )
        return
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster)
    rollout = RealworldEvalRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=placement.get_strategy("rollout"),
    )
    print(rollout.init_worker().wait(), flush=True)
    env = RealworldEvalEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    env.init_worker().wait()
    observations = Channel.create("RealEvalObs", distributed=True)
    actions = Channel.create("RealEvalActions", distributed=True)
    control = Channel.create("RealEvalControl", distributed=True)
    serving = rollout.serve(observations, actions)
    records = []
    active = None
    interrupted = False
    try:
        for episode_id in range(1, cfg.evaluation.num_episodes + 1):
            active = env.run_episode(episode_id, observations, actions, control)
            record = active.wait()[0]
            active = None
            record["checkpoint"] = cfg.runner.ckpt_path
            records.append(record)
            append_record(root / "episodes.jsonl", record)
            print(
                json.dumps(
                    {
                        "episode": record,
                        "summary": summarize(records, cfg.evaluation.num_episodes),
                    }
                ),
                flush=True,
            )
            if not record["completed"]:
                raise RuntimeError(f"Evaluation stopped: {record}")
    except KeyboardInterrupt:
        interrupted = True
        control.put(True, key="cancel")
        print(
            "Cancellation requested; stop issuing actions at the next control boundary. This is not an emergency stop.",
            flush=True,
        )
        if active is not None:
            record = active.wait()[0]
            records.append(record)
            append_record(root / "episodes.jsonl", record)
    finally:
        observations.put({"stop": True}, key="obs")
        serving.wait()
        env.finish_evaluation().wait()
        summary = summarize(records, cfg.evaluation.num_episodes)
        summary.update(
            checkpoint=cfg.runner.ckpt_path,
            policy_source=(
                "checkpoint" if cfg.runner.ckpt_path else "initial_residual"
            ),
            policy_mode=cfg.evaluation.policy_mode,
            cancelled=interrupted,
        )
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
