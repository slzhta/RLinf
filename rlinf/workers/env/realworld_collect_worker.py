"""Human-reviewed collection, separate from the standalone evaluation worker."""

import asyncio
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.utils.realworld_collection import EpisodeCapture
from rlinf.utils.realworld_eval import (
    append_record,
    evaluation_observation,
    run_episode,
)
from rlinf.workers.env.realworld_eval_worker import RealworldEvalEnvWorker


class RealworldCollectEnvWorker(RealworldEvalEnvWorker):
    def run_episode(self, episode_id, observations, actions, control):
        if getattr(self, "pending_capture", None) is not None:
            raise RuntimeError("Review the previous episode before starting another")
        env = self.eval_env_list[0]
        root = Path(os.environ["RLINF_LOG_PATH"]) / self.cfg.evaluation.run_relative_dir
        result = dict(
            episode_id=episode_id,
            started_at=time.time(),
            completed=False,
            success=False,
            status="error",
        )
        capture = EpisodeCapture()
        first_step = True

        def check_cancel():
            try:
                control.get_nowait(key="cancel")
            except asyncio.QueueEmpty:
                return
            raise InterruptedError("Collection cancelled; no further actions sent")

        def reset():
            check_cancel()
            self.begin_rollout_video(mode="eval")
            obs, info = env.reset()
            capture.observation(obs)
            return obs, info

        def predict(obs):
            nonlocal first_step
            check_cancel()
            observations.put(
                {
                    "obs": evaluation_observation(
                        obs,
                        self.cfg.rollout.get("residual_base_inference", False),
                        first_step,
                    )
                },
                key="obs",
            )
            timeout = float(self.cfg.evaluation.get("action_timeout_seconds", 30))
            deadline = time.monotonic() + timeout
            while True:
                check_cancel()
                try:
                    response = actions.get_nowait(key="actions")
                    break
                except asyncio.QueueEmpty:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"No policy response in {timeout:g}s")
                    time.sleep(0.01)
            check_cancel()
            if "error" in response:
                raise RuntimeError(response["error"])
            first_step = False
            action = prepare_actions(
                raw_chunk_actions=response["actions"],
                env_type=self.cfg.env.eval.env_type,
                model_type=self.cfg.actor.model.model_type,
                num_action_chunks=1,
                action_dim=self.cfg.actor.model.action_dim,
                policy=self.cfg.actor.model.get("policy_setup", None),
                wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            )
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite action; refusing robot command")
            return action

        def step(action):
            check_cancel()
            recorded = np.array(action, copy=True)
            started = time.time()
            obs, reward, terminated, truncated, infos = env.chunk_step(action)
            capture.transition(
                recorded, reward, terminated, truncated, obs[-1], started
            )
            return obs[-1], reward, terminated, truncated, infos[-1]

        try:
            result.update(
                run_episode(reset, step, predict, self.cfg.env.eval.max_episode_steps)
            )
        except Exception as exc:
            result.update(
                status="interrupted" if isinstance(exc, InterruptedError) else "error",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            result["elapsed_seconds"] = time.time() - result["started_at"]
            if isinstance(env, RecordVideo):
                result["video_path"] = str(
                    root / "video/real" / f"seed_{env.seed}" / f"{env.video_cnt}.mp4"
                )
                env.flush_video()
            append_record(root / "episodes.jsonl", result)
        if result["completed"]:
            self.pending_capture = (episode_id, capture, result, root)
        return result

    def review_episode(self, episode_id, keep):
        pending = getattr(self, "pending_capture", None)
        if pending is None or pending[0] != episode_id:
            raise ValueError("No matching pending episode")
        _, capture, result, root = pending
        record = dict(
            episode_id=episode_id, accepted=bool(keep), success=result["success"]
        )
        if keep:
            record["data_path"] = capture.save(
                root / "data" / f"episode_{episode_id:06d}",
                result,
                OmegaConf.to_container(self.cfg, resolve=True),
            )
        append_record(root / "collection_decisions.jsonl", record)
        self.pending_capture = None
        return record

    def finish_evaluation(self):
        self.pending_capture = None
        return super().finish_evaluation()
