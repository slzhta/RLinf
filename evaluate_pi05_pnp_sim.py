import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from omegaconf import OmegaConf

import websockets.sync.client
from openpi_client import websocket_client_policy

import mani_skill
import rlinf.envs.maniskill.tasks.digital_twin.pick_and_place


def to_bool(value):
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def create_environment():
    config_path = Path(
        "examples/embodiment/config/env/"
        "maniskill_pick_and_place_co_rl.yaml"
    )

    cfg = OmegaConf.load(config_path)

    # 避免独立加载 YAML 时引用训练配置中的插值变量。
    cfg.init_params.max_episode_steps = 120

    params = OmegaConf.to_container(
        cfg.init_params,
        resolve=True,
    )

    # 此字段属于 RLinf 逻辑，不是 ManiSkill BaseEnv 参数。
    params.pop("use_sparse_reward", None)

    params["num_envs"] = 1
    params["render_mode"] = "rgb_array"
    params["sim_backend"] = "gpu"

    return gym.make(**params)


def create_client(host, port):
    # OpenPI 第一次推理可能需要较长时间。关闭 WebSocket ping，
    # 避免模型仍在初始化时被误判为连接超时。
    original_connect = websockets.sync.client.connect

    def connect_without_ping(*args, **kwargs):
        kwargs["ping_interval"] = None
        kwargs["close_timeout"] = 60
        kwargs["max_size"] = None
        kwargs["compression"] = None
        return original_connect(*args, **kwargs)

    websockets.sync.client.connect = connect_without_ping

    return websocket_client_policy.WebsocketClientPolicy(
        host=host,
        port=port,
    )


def make_model_observation(info):
    extracted = info["extracted_obs"]

    main_image = (
        extracted["main_images"][0]
        .detach()
        .cpu()
        .numpy()
    )

    wrist_image = (
        extracted["extra_view_images"][0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    state = (
        extracted["states"][0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    prompt = extracted["task_descriptions"][0]

    return {
        "observation/image": main_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/pi05_pnp_sim_eval/results.json"),
    )
    args = parser.parse_args()

    print("Connecting to OpenPI server...", flush=True)
    client = create_client(args.host, args.port)
    print("Connected.", flush=True)

    print("Creating ManiSkill environment...", flush=True)
    env = create_environment()
    print("Environment ready.", flush=True)

    results = []

    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        _, info = env.reset(seed=episode_seed)

        success = False
        failed = False
        grasped = False
        lifted = False
        placed = False
        dropped = False
        executed_steps = 0
        inference_calls = 0

        start_time = time.time()

        while executed_steps < 120:
            model_observation = make_model_observation(info)

            print(
                f"[episode {episode + 1}/{args.episodes}] "
                f"inference {inference_calls + 1}, "
                f"step {executed_steps}",
                flush=True,
            )

            output = client.infer(model_observation)
            actions = np.asarray(
                output["actions"],
                dtype=np.float32,
            )

            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]

            if actions.ndim != 2:
                raise RuntimeError(
                    f"Unexpected actions shape: {actions.shape}"
                )

            if actions.shape[1] < 7:
                raise RuntimeError(
                    f"Expected at least 7 action dimensions, "
                    f"got {actions.shape}"
                )

            actions = actions[:, :7]

            if not np.isfinite(actions).all():
                raise RuntimeError("Model produced NaN or Inf actions")

            # 数据集动作范围为 [-1, 1]。
            actions = np.clip(actions, -1.0, 1.0)

            inference_calls += 1

            for action in actions:
                action_tensor = torch.as_tensor(
                    action[None, :],
                    dtype=torch.float32,
                    device="cuda:0",
                )

                _, _, terminated, truncated, info = env.step(
                    action_tensor
                )

                executed_steps += 1

                success = success or to_bool(info["success"])
                failed = failed or to_bool(info["fail"])
                grasped = grasped or to_bool(info["grasp_once"])
                lifted = lifted or to_bool(info["lift_once"])
                placed = placed or to_bool(info["place_once"])
                dropped = dropped or to_bool(info["drop_once"])

                is_terminated = to_bool(terminated)
                is_truncated = to_bool(truncated)

                if (
                    success
                    or failed
                    or is_terminated
                    or is_truncated
                    or executed_steps >= 120
                ):
                    break

            if success or failed or executed_steps >= 120:
                break

        elapsed = time.time() - start_time

        result = {
            "episode": episode,
            "seed": episode_seed,
            "success": success,
            "fail": failed,
            "grasp_once": grasped,
            "lift_once": lifted,
            "place_once": placed,
            "drop_once": dropped,
            "steps": executed_steps,
            "inference_calls": inference_calls,
            "elapsed_seconds": elapsed,
        }
        results.append(result)

        completed = len(results)

        print(
            f"\nEpisode {episode + 1}: "
            f"success={success}, "
            f"grasp={grasped}, "
            f"lift={lifted}, "
            f"place={placed}, "
            f"drop={dropped}, "
            f"steps={executed_steps}, "
            f"time={elapsed:.1f}s"
        )

        print(
            "Current rates: "
            f"success={sum(x['success'] for x in results) / completed:.2%}, "
            f"grasp={sum(x['grasp_once'] for x in results) / completed:.2%}, "
            f"lift={sum(x['lift_once'] for x in results) / completed:.2%}, "
            f"place={sum(x['place_once'] for x in results) / completed:.2%}"
        )
        print()

    total = len(results)

    summary = {
        "episodes": total,
        "successes": sum(x["success"] for x in results),
        "success_rate": sum(x["success"] for x in results) / total,
        "grasp_rate": sum(x["grasp_once"] for x in results) / total,
        "lift_rate": sum(x["lift_once"] for x in results) / total,
        "place_rate": sum(x["place_once"] for x in results) / total,
        "drop_rate": sum(x["drop_once"] for x in results) / total,
        "mean_steps": float(
            np.mean([x["steps"] for x in results])
        ),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print("=" * 60)
    print("FINAL EVALUATION")
    print(f"Episodes:     {summary['episodes']}")
    print(f"Successes:    {summary['successes']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Grasp rate:   {summary['grasp_rate']:.2%}")
    print(f"Lift rate:    {summary['lift_rate']:.2%}")
    print(f"Place rate:   {summary['place_rate']:.2%}")
    print(f"Drop rate:    {summary['drop_rate']:.2%}")
    print(f"Mean steps:   {summary['mean_steps']:.1f}")
    print(f"Saved to:     {args.output}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
