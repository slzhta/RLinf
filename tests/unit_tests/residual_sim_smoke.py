"""Manual simulation and FSDP smoke check for the a4 residual fixes."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_residual_policy import model_config
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models.embodiment.modules.resnet_utils import ResNet10
from rlinf.models.embodiment.residual_policy.residual_policy import ResidualPolicy
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg


def main():
    root = Path(__file__).resolve().parents[2]
    os.environ["EMBODIED_PATH"] = str(root / "examples/embodiment")
    with initialize_config_dir(
        config_dir=str(root / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_a4")
    OmegaConf.resolve(cfg)
    validate_residual_cfg(cfg)
    assert cfg.actor.global_batch_size % cfg.actor.micro_batch_size == 0
    assert (
        cfg.env.train.total_num_envs * cfg.env.train.max_steps_per_rollout_epoch
    ) % cfg.actor.global_batch_size == 0
    assert cfg.actor.model.initial_logstd == 0.0
    assert Path(cfg.base_model.model_path).is_dir()
    assert (
        Path(cfg.actor.model.model_path) / cfg.actor.model.encoder_config.ckpt_name
    ).is_file()
    env = ManiskillEnv(cfg.env.train, 2, 0, 2, None)
    env.reset()

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense reward path must never execute")

    env.env.unwrapped.compute_dense_reward = forbidden
    for _ in range(8):
        _, reward, _, _, _ = env.step(torch.zeros(2, 7, device=env.device))
        assert torch.all((reward == 0) | (reward == 1))
    reward = env.env.unwrapped.get_reward(
        None,
        None,
        {
            "success": torch.tensor([False, True], device=env.device),
            "fail": torch.tensor([True, False], device=env.device),
        },
    )
    torch.testing.assert_close(reward, torch.tensor([0.0, 1.0], device=env.device))
    env.env.close()
    print("SIM_SPARSE_DISPATCH_PASS", flush=True)

    with tempfile.TemporaryDirectory(prefix="residual-fsdp-") as tmp:
        torch.distributed.init_process_group(
            "nccl", init_method=f"file://{tmp}/rendezvous", rank=0, world_size=1
        )
        path = Path(tmp)
        torch.save(ResNet10().state_dict(), path / "resnet10_pretrained.pt")
        model_cfg = model_config(path)
        model_cfg.logstd_range = [-5.0, 0.0]
        policy = ResidualPolicy(model_cfg).cuda()
        model = FullyShardedDataParallel(
            policy,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=True,
            device_id=0,
        )
        with torch.no_grad():
            policy.actor_logstd.fill_(0.3)
        FSDPModelManager._project_policy_parameters(SimpleNamespace(model=model))
        assert torch.all(policy.actor_logstd == 0)
        print("FSDP_PROJECTION_PASS", flush=True)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
