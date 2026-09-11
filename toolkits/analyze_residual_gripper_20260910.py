"""Analyze a saved residual policy with paired simulation ablations, without Ray.

Use the run's saved config, an immutable full_weights.pt, and a spare GPU selected
by CUDA_VISIBLE_DEVICES. Never modifies the training configuration or weights.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from rlinf.envs.maniskill.residual_maniskill_env import ResidualManiskillEnv
from rlinf.models.embodiment.residual_policy import get_model
from rlinf.workers.env.env_worker import EnvWorker


def statistics(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles_01_05_25_50_75_95_99": np.quantile(
            values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        ).tolist(),
    }


def target(command, previous):
    return torch.where(
        command >= 0.5,
        torch.ones_like(previous),
        torch.where(command <= -0.5, -torch.ones_like(previous), previous),
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["full_mean", "no_gripper", "base_only", "full_sample"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(args.run / "tensorboard/config.yaml")
    checkpoints = args.run / cfg.runner.logger.experiment_name / "checkpoints"
    weights_path = (
        checkpoints
        / f"global_step_{args.step}"
        / "actor/model_state_dict/full_weights.pt"
    )
    checkpoint_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    policy = get_model(cfg.actor.model).cuda().eval()
    policy.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True), strict=True
    )
    for param in policy.parameters():
        param.requires_grad_(False)
    std = policy._action_std(policy.actor_logstd).detach()
    history = {}
    for path in sorted(
        checkpoints.glob("global_step_*/actor/model_state_dict/full_weights.pt")
    ):
        weights = torch.load(path, map_location="cpu", weights_only=True)
        logstd = weights["actor_logstd"].flatten()
        history[path.parents[2].name] = {
            "raw_logstd": logstd.tolist(),
            "clamped_std": logstd.clamp(*cfg.actor.model.logstd_range).exp().tolist(),
            "mean_head_bias": weights["actor_mean.bias"].tolist(),
        }
        del weights
    metadata = {
        "run": str(args.run),
        "checkpoint": str(weights_path),
        "checkpoint_sha256": checkpoint_hash,
        "initial_logstd": list(cfg.actor.model.initial_logstd),
        "effective_std": std.cpu().tolist(),
        "scale": list(cfg.actor.model.residual_action_scale),
        "seeds": args.seeds,
        "num_envs_per_seed": args.num_envs,
        "steps": args.steps,
        "checkpoint_history": history,
        "conditions": args.conditions,
        "device": torch.cuda.get_device_name(),
        "scope": "Fixed initial episodes, no auto-reset; ignore all transitions after first done. No Ray or training changes.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print("METADATA", json.dumps(metadata), flush=True)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.eval, resolve=True))
    injector = SimpleNamespace(cfg=cfg)
    injector._translate_alignment_for_maniskill = lambda c: (
        EnvWorker._translate_alignment_for_maniskill(None, c)
    )
    EnvWorker._inject_sim_real_alignment_cfg(injector, env_cfg)
    EnvWorker._inject_sim_real_reward_cfg(injector, env_cfg)
    env_cfg.auto_reset = False
    env_cfg.video_cfg.save_video = False
    env = ResidualManiskillEnv(env_cfg, args.num_envs, 0, 1, None)
    OmegaConf.save(env_cfg, args.output / "effective_env.yaml")
    # Load before reseeding, so model initialization cannot perturb the first paired rollout.
    env._load_base()
    scale = env.residual_scale
    summary = {}
    starts = {}
    for condition in args.conditions:
        condition_records = []
        episodes = []
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            obs, _ = env.reset(seed=seed)
            initial = obs["states"].cpu().numpy().copy()
            image_start = obs["main_images"].cpu().numpy().copy()
            if seed not in starts:
                starts[seed] = (initial, image_start)
            assert np.allclose(initial, starts[seed][0], atol=1e-5), (
                "Unpaired robot initial state"
            )
            print(
                "INITIAL_PAIR",
                condition,
                seed,
                "image_mean_abs_difference",
                float(np.abs(image_start.astype(float) - starts[seed][1]).mean()),
                flush=True,
            )
            generator = torch.Generator(device=env.device).manual_seed(10000 + seed)
            active = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
            success = torch.zeros_like(active)
            first_success = torch.full((args.num_envs,), -1, device=env.device)
            records = {
                key: []
                for key in (
                    "active",
                    "base",
                    "mean",
                    "raw",
                    "executed",
                    "before",
                    "after",
                    "base_target",
                    "success",
                    "grasp",
                    "lift",
                    "drop",
                )
            }
            video = imageio.get_writer(
                str(args.output / f"{condition}_seed{seed}.mp4"), fps=10
            )
            try:
                with torch.inference_mode():
                    for step in range(args.steps):
                        base = env.base_cache.actions[
                            torch.arange(args.num_envs, device=env.device),
                            env.base_cache.position,
                        ].clone()
                        before = obs["states"][:, -1].clone()
                        mean, _ = policy.predict_action_batch(obs, mode="eval")
                        mean = mean[:, 0]
                        raw = mean.clone()
                        if condition == "full_sample":
                            raw += (
                                torch.randn(
                                    raw.shape, generator=generator, device=raw.device
                                )
                                * std
                            )
                        elif condition == "no_gripper":
                            raw[:, 6] = 0
                        elif condition == "base_only":
                            raw.zero_()
                        elif condition != "full_mean":
                            raise ValueError(condition)
                        expected = (base + scale * raw.clamp(-1, 1)).clamp(-1, 1)
                        base_target = target(base[:, 6].clamp(-1, 1), before)
                        obs, reward, terminated, truncated, info = env.step(
                            raw, auto_reset=False
                        )
                        after = obs["states"][:, -1].clone()
                        torch.testing.assert_close(
                            expected, info["residual_executed_action"]
                        )
                        torch.testing.assert_close(
                            after, target(expected[:, 6], before)
                        )
                        current_success = info["success"].bool()
                        first_success[
                            (first_success < 0) & active & current_success
                        ] = step
                        success |= active & current_success
                        values = {
                            "active": active,
                            "base": base[:, 6],
                            "mean": mean[:, 6],
                            "raw": raw[:, 6],
                            "executed": expected[:, 6],
                            "before": before,
                            "after": after,
                            "base_target": base_target,
                            "success": current_success,
                        }
                        for key, source in [
                            ("grasp", "is_src_obj_grasped"),
                            ("lift", "lift_once"),
                            ("drop", "drop_once"),
                        ]:
                            values[key] = info.get(source, torch.zeros_like(active))
                        for key, value in values.items():
                            records[key].append(value.detach().cpu().numpy().copy())
                        frames = obs["main_images"][:4].cpu().numpy()
                        rendered = []
                        for idx, frame in enumerate(frames):
                            im = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
                            draw = ImageDraw.Draw(im)
                            draw.rectangle((0, 0, im.width, 40), fill="black")
                            draw.text(
                                (3, 2),
                                f"{condition} e{idx} t{step} b={base[idx, 6]:+.2f}",
                                fill="white",
                            )
                            draw.text(
                                (3, 17),
                                f"r={raw[idx, 6]:+.2f} g={expected[idx, 6]:+.2f} state={after[idx]:+.0f}",
                                fill="white",
                            )
                            rendered.append(np.asarray(im))
                        video.append_data(
                            np.concatenate(
                                [
                                    np.concatenate(rendered[:2], axis=1),
                                    np.concatenate(rendered[2:4], axis=1),
                                ],
                                axis=0,
                            )
                        )
                        active &= ~(terminated.bool() | truncated.bool())
                        if (step + 1) % 20 == 0:
                            print(
                                "PROGRESS",
                                condition,
                                seed,
                                step + 1,
                                "success",
                                int(success.sum()),
                                "active",
                                int(active.sum()),
                                flush=True,
                            )
                        if not active.any():
                            break
            finally:
                video.close()
            arrays = {key: np.stack(value) for key, value in records.items()}
            np.savez_compressed(args.output / f"{condition}_seed{seed}.npz", **arrays)
            condition_records.append(arrays)
            for idx in range(args.num_envs):
                episodes.append(
                    {
                        "seed": seed,
                        "env": idx,
                        "success": bool(success[idx]),
                        "first_success_step": int(first_success[idx]),
                        "valid_steps": int(arrays["active"][:, idx].sum()),
                        "gripper_switches": int(
                            (
                                (arrays["before"][:, idx] != arrays["after"][:, idx])
                                & arrays["active"][:, idx]
                            ).sum()
                        ),
                    }
                )
            print(
                "ROLLOUT_DONE",
                condition,
                seed,
                int(success.sum()),
                "/",
                args.num_envs,
                flush=True,
            )
        pooled = {
            key: np.concatenate([a[key][a["active"]] for a in condition_records])
            for key in condition_records[0]
            if key != "active"
        }
        base_close = (pooled["base_target"] < 0) & (pooled["before"] > 0)
        base_open = (pooled["base_target"] > 0) & (pooled["before"] < 0)
        summary[condition] = {
            "episodes": episodes,
            "success_count": sum(e["success"] for e in episodes),
            "episode_count": len(episodes),
            "valid_transitions": len(pooled["raw"]),
            "raw_residual": statistics(pooled["raw"]),
            "policy_mean": statistics(pooled["mean"]),
            "base_gripper": statistics(pooled["base"]),
            "clip_fraction": float((np.abs(pooled["raw"]) >= 1).mean()),
            "positive_fraction": float((pooled["raw"] > 0).mean()),
            "override_steps": int((pooled["base_target"] != pooled["after"]).sum()),
            "base_close_opportunities": int(base_close.sum()),
            "suppressed_close_steps": int((base_close & (pooled["after"] > 0)).sum()),
            "base_open_opportunities": int(base_open.sum()),
            "suppressed_open_steps": int((base_open & (pooled["after"] < 0)).sum()),
            "closed_hold_reopen_steps": int(
                (
                    (pooled["before"] < 0)
                    & (pooled["base_target"] < 0)
                    & (pooled["after"] > 0)
                ).sum()
            ),
            "open_hold_close_steps": int(
                (
                    (pooled["before"] > 0)
                    & (pooled["base_target"] > 0)
                    & (pooled["after"] < 0)
                ).sum()
            ),
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(
            "CONDITION_DONE",
            condition,
            summary[condition]["success_count"],
            "/",
            len(episodes),
            flush=True,
        )
    env.env.close()
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == checkpoint_hash
    print("ANALYSIS_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
