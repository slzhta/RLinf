"""Episode accounting for standalone real-robot evaluation (no training)."""

import json
import os
from pathlib import Path
from typing import Any, Callable


def evaluation_checkpoint(cfg: Any) -> str | None:
    """Reject missing checkpoints unless residual initialization is explicit."""
    checkpoint = cfg.runner.get("ckpt_path")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint missing: {path}")
        return str(path)
    if (
        cfg.evaluation.get("allow_initial_residual", False)
        and cfg.actor.model.model_type == "residual_policy"
        and cfg.rollout.get("residual_base_inference", False)
    ):
        return None
    raise ValueError("Evaluation requires a checkpoint or explicit initial residual")


def evaluation_observation(obs: dict, residual: bool, first_step: bool) -> dict:
    """Invalidate the frozen base action cache exactly at episode starts."""
    if not residual:
        return obs
    import numpy as np

    return {
        **obs,
        "_residual_reset_mask": np.full(
            (obs["main_images"].shape[0],), first_step, dtype=np.bool_
        ),
    }


def scalar(value: Any) -> Any:
    """Read one scalar from a single-environment array or tensor."""
    if hasattr(value, "reshape"):
        return value.reshape(-1)[0].item()
    return value


def make_peg_z_guard(cfg: Any) -> Callable | None:
    """Build the opt-in RLPD eval guard in robot-base coordinates."""
    threshold = cfg.evaluation.get("stop_at_target_z_distance", None)
    if threshold is None:
        return None
    import numpy as np
    from scipy.spatial.transform import Rotation

    env = cfg.env.eval
    if (
        not cfg.runner.only_eval
        or cfg.actor.model.model_type != "cnn_policy"
        or cfg.algorithm.loss_type != "embodied_sac"
        or env.init_params.id != "FrankaCoTrainingPegInsertionEnv-v1"
        or list(env.state_keys) != ["ee_target_delta"]
        or cfg.actor.model.num_action_chunks != 1
        or env.auto_reset
        or env.total_num_envs != 1
    ):
        raise ValueError("Target-z guard requires single-environment CNN-RLPD Peg eval")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 < threshold <= 0.01:
        raise ValueError("Target-z guard threshold must be in (0, 0.01] metres")
    target = np.asarray(env.override_cfg.peg_config.target_ee_pose, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Target-z guard requires a finite six-dimensional target pose")
    base_z_row = Rotation.from_euler("xyz", target[3:]).as_matrix()[2]

    def guard(obs):
        state = obs["states"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=float)
        if state.shape != (1, 6) or not np.isfinite(state).all():
            raise ValueError("Invalid measured Peg state; refusing another action")
        z_delta = float(base_z_row @ state[0, :3])
        if abs(z_delta) <= threshold + 1e-9:
            return {
                "guard_z_delta_m": z_delta,
                "guard_threshold_m": threshold,
                "manual_scoring_required": True,
            }
        return None

    return guard


def run_episode(
    reset: Callable,
    step: Callable,
    predict: Callable,
    max_steps: int,
    stop_guard: Callable | None = None,
) -> dict:
    """Stop at the first terminal transition; never reset after it."""
    obs, _ = reset()
    success = False
    intervened = False
    reward_sum = 0.0
    guard_info = stop_guard(obs) if stop_guard is not None else None
    if guard_info is not None:
        return {
            "steps": 0,
            "success": False,
            "intervened": False,
            "status": "z_guard",
            "completed": True,
            "return": 0.0,
            **guard_info,
        }
    for index in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = step(predict(obs))
        episode = info.get("episode", {})
        if "success_once" not in episode and "success" not in info:
            raise ValueError("Environment must expose explicit episode success")
        success |= bool(scalar(episode.get("success_once", info.get("success", False))))
        intervened |= bool(scalar(episode.get("intervened_once", False)))
        aborted = bool(
            scalar(episode.get("aborted", info.get("discard_trajectory", False)))
        )
        reward_sum += float(scalar(reward))
        guard_info = stop_guard(obs) if stop_guard is not None else None
        if (
            aborted
            or guard_info is not None
            or scalar(terminated)
            or scalar(truncated)
            or index == max_steps
        ):
            reason = (
                "aborted"
                if aborted
                else "z_guard"
                if guard_info is not None
                else "success"
                if success
                else "timeout"
                if scalar(truncated) or index == max_steps
                else "failure"
            )
            return {
                "steps": index,
                "success": success and not aborted,
                "intervened": intervened,
                "status": reason,
                "completed": not aborted,
                "return": reward_sum,
                **(guard_info or {}),
            }
    raise ValueError("max_steps must be positive")


def summarize(records: list[dict], requested: int) -> dict:
    """Count real completed episodes, retaining aborts separately."""
    completed = [r for r in records if r.get("completed", False)]
    successes = sum(bool(r["success"]) for r in completed)
    report = {
        "requested": requested,
        "attempts": len(records),
        "completed": len(completed),
        "successes": successes,
        "failures": len(completed) - successes,
        "interrupted": len(records) - len(completed),
        "success_rate": successes / len(completed) if completed else None,
        "successes_per_attempt": successes / len(records) if records else None,
        "finished": len(completed) == requested,
    }
    if any(r.get("manual_scoring_required", False) for r in records):
        report["manual_scoring_required"] = True
        report["z_guard_stops"] = sum(r.get("status") == "z_guard" for r in records)
    return report


def append_record(path: Path, record: dict) -> None:
    """Persist each result immediately so interruption retains earlier trials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
