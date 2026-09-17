"""Pour SpaceMouse collection using the native real-world environment."""

import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from rlinf.data.lerobot_writer import LeRobotDatasetWriter
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Worker
from rlinf.utils.pour_collection import export_success, validate_observation
from rlinf.utils.realworld_collection import EpisodeCapture, numpy_copy
from rlinf.utils.realworld_eval import append_record


class PourCollectionWorker(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        root = Path(self.cfg.runner.logger.log_path)
        env = RealWorldEnv(self.cfg.env.eval, 1, 0, 1, self.worker_info)
        writer = None
        successes, attempt = 0, 0
        try:
            writer = LeRobotDatasetWriter(
                str(root / "lerobot"),
                robot_type="panda",
                fps=10,
                image_shape=(224, 224, 3),
                state_dim=14,
                action_dim=6,
                has_wrist_image=True,
                has_extra_view_image=False,
                use_incremental_stats=True,
                stats_sample_ratio=1.0,
            )
            while successes < self.cfg.runner.num_data_episodes:
                attempt += 1
                capture = EpisodeCapture()
                obs, _ = env.reset()
                validate_observation(obs)
                capture.observation(obs)
                discarded = False
                for step in range(self.cfg.env.eval.max_episode_steps):
                    commanded = np.zeros((1, 6), dtype=np.float32)
                    started = time.time()
                    obs, reward, terminated, truncated, info = env.step(commanded)
                    action = numpy_copy(
                        info.get("intervene_action", commanded)
                    ).reshape(1, 6)
                    validate_observation(obs)
                    capture.transition(
                        action, reward, terminated, truncated, obs, started
                    )
                    discarded |= bool(
                        numpy_copy(info.get("discard_trajectory", False)).any()
                    )
                    term = bool(numpy_copy(terminated).any())
                    trunc = bool(numpy_copy(truncated).any())
                    if term or trunc or step + 1 == self.cfg.env.eval.max_episode_steps:
                        success = (
                            bool(numpy_copy(info.get("success", False)).any())
                            and not discarded
                        )
                        record = dict(
                            episode_id=attempt,
                            completed=True,
                            steps=step + 1,
                            success=success,
                            discarded=discarded,
                            terminated=term,
                            truncated=trunc,
                            finished_at=time.time(),
                        )
                        raw = capture.save(
                            root / "raw" / f"episode_{attempt:06d}",
                            record,
                            OmegaConf.to_container(self.cfg, resolve=True),
                        )
                        if success:
                            export_success(writer, capture.arrays(record))
                            successes += 1
                        append_record(
                            root / "attempts.jsonl",
                            {**record, "raw": raw, "successes": successes},
                        )
                        self.log_info(
                            f"Pour collection: {successes}/{self.cfg.runner.num_data_episodes} successes; attempt={attempt}"
                        )
                        break
        finally:
            try:
                if writer is not None:
                    writer.finalize()
            finally:
                env.close()
