"""Inference-only real collection with per-episode terminal approval."""

import json
import sys
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from eval_realworld import validate_eval
from rlinf.config import validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.realworld_collection import ask_disposition
from rlinf.utils.realworld_eval import append_record, summarize
from rlinf.workers.env.realworld_collect_worker import RealworldCollectEnvWorker
from rlinf.workers.rollout.hf.realworld_eval_worker import RealworldEvalRolloutWorker


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="real_eval_peg_insertion_openpi_residual_wrist_state",
)
def main(cfg):
    source = hydra.compose(config_name=cfg.evaluation.training_config)
    cfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(source)), cfg)
    validate_eval(cfg)
    if not cfg.evaluation.check_only and not sys.stdin.isatty():
        raise ValueError("Collection needs an interactive terminal for keep/discard")
    root = Path(cfg.runner.logger.log_path)
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "config.yaml", resolve=True)
    if cfg.evaluation.check_only:
        print("Collection config validated; no Ray or robot initialized.", flush=True)
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
    env = RealworldCollectEnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=placement.get_strategy("env"),
    )
    env.init_worker().wait()
    observations = Channel.create("RealCollectObs", distributed=True)
    actions = Channel.create("RealCollectActions", distributed=True)
    control = Channel.create("RealCollectControl", distributed=True)
    serving = rollout.serve(observations, actions)
    records, decisions = [], []
    active = None
    stopped = False
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
                raise RuntimeError(f"Collection stopped: {record}")
            choice = ask_disposition(record)
            decision = env.review_episode(episode_id, choice == "k").wait()[0]
            decisions.append(decision)
            append_record(root / "collection_decisions.jsonl", decision)
            print(json.dumps(decision), flush=True)
            if choice == "q":
                stopped = True
                break
    except (KeyboardInterrupt, EOFError):
        stopped = True
        control.put(True, key="cancel")
        print(
            "Stopping at the next control boundary; this is not an emergency stop.",
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
            policy_mode=cfg.evaluation.policy_mode,
            stopped=stopped,
            accepted=sum(d["accepted"] for d in decisions),
            reviewed=len(decisions),
        )
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
