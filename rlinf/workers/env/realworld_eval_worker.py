"""Explicit episode lifecycle for one real robot, isolated from training."""

import asyncio
import os
import time
from pathlib import Path

import numpy as np

from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.wrappers import RecordVideo
from rlinf.utils.realworld_eval import (
    append_record,
    evaluation_observation,
    make_peg_z_guard,
    run_episode,
)
from rlinf.workers.env.env_worker import EnvWorker


class RealworldEvalEnvWorker(EnvWorker):
    def run_episode(self, episode_id, observations, actions, control):
        """Run one episode and durably journal its result on the NUC."""
        env = self.eval_env_list[0]
        stop_guard = make_peg_z_guard(self.cfg)
        started = time.time()
        result = {
            "episode_id": episode_id,
            "started_at": started,
            "completed": False,
            "success": False,
            "status": "error",
        }
        root = Path(os.environ["RLINF_LOG_PATH"]) / self.cfg.evaluation.run_relative_dir
        first_step = True

        def check_cancel():
            try:
                control.get_nowait(key="cancel")
            except asyncio.QueueEmpty:
                return
            raise InterruptedError("Evaluation cancelled; no further actions sent")

        def reset():
            nonlocal first_step
            check_cancel()
            first_step = True
            self.begin_rollout_video(mode="eval")
            if (
                stop_guard is not None
                and getattr(self, "_guard_reset_observation", None) is not None
            ):
                observation = self._guard_reset_observation
                self._guard_reset_observation = None
                return observation
            return env.reset()

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
                        raise TimeoutError(
                            f"No policy response in {timeout:g}s; stopping evaluation"
                        )
                    time.sleep(0.01)
            check_cancel()
            if "error" in response:
                raise RuntimeError(response["error"])
            first_step = False
            raw = response["actions"]
            prepared = prepare_actions(
                raw_chunk_actions=raw,
                env_type=self.cfg.env.eval.env_type,
                model_type=self.cfg.actor.model.model_type,
                num_action_chunks=1,
                action_dim=self.cfg.actor.model.action_dim,
                policy=self.cfg.actor.model.get("policy_setup", None),
                wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
            )
            if not np.isfinite(prepared).all():
                raise ValueError("Non-finite action; refusing robot command")
            return prepared

        def step(action):
            check_cancel()
            obs, rewards, terminated, truncated, infos = env.chunk_step(action)
            return obs[-1], rewards, terminated, truncated, infos[-1]

        try:
            result.update(
                run_episode(
                    reset,
                    step,
                    predict,
                    self.cfg.env.eval.max_episode_steps,
                    stop_guard=stop_guard,
                )
            )
        except Exception as exc:
            result.update(
                status="interrupted" if isinstance(exc, InterruptedError) else "error",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            result["elapsed_seconds"] = time.time() - started
            if isinstance(env, RecordVideo):
                video_id = env.video_cnt
                result["video_path"] = str(
                    root / "video/real" / f"seed_{env.seed}" / f"{video_id}.mp4"
                )
                env.flush_video()
            if stop_guard is not None:
                result["manual_scoring_required"] = True
            if result["status"] == "z_guard":
                try:
                    check_cancel()
                    self._guard_reset_observation = env.reset()
                    result["guard_reset_completed"] = True
                except Exception as exc:
                    result.update(
                        completed=False,
                        guard_reset_completed=False,
                        status="error",
                        error=f"Guard reset failed: {type(exc).__name__}: {exc}",
                    )
            append_record(root / "episodes.jsonl", result)
        return result

    def finish_evaluation(self):
        """Drain video writes without issuing a reset or recovery command."""
        for env in self.eval_env_list:
            if isinstance(env, RecordVideo):
                env._executor.shutdown(wait=True)
        return True
