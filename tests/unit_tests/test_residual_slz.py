"""Residual adapter contracts against the unchanged SLZ CNN/PPO implementation."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.envs.residual import BaseActionCache
from rlinf.models.embodiment.modules.resnet_utils import ResNet10
from rlinf.models.embodiment.residual_policy.residual_policy import (
    ResidualConfig,
    ResidualPolicy,
)
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg
from rlinf.utils.utils import masked_mean, reshape_entropy, reshape_entropy_mask


@pytest.fixture(scope="module")
def policy(tmp_path_factory):
    torch.set_num_threads(2)
    path = tmp_path_factory.mktemp("encoder")
    torch.save(ResNet10().state_dict(), path / "resnet10_pretrained.pt")
    cfg = ResidualConfig()
    cfg.update_from_dict({
        "model_path": str(path),
        "encoder_config": {"ckpt_name": "resnet10_pretrained.pt", "dropout": 0.0},
        "use_state": True, "state_dim": 14, "image_size": [3, 128, 128],
        "image_num": 2, "action_dim": 7, "base_horizon": 10,
        "add_value_head": True, "initial_logstd": -1.0,
        "logstd_range": [-5.0, 0.0],
    })
    return ResidualPolicy(cfg).eval()


def test_high_resolution_inputs_and_ppo_replay(policy):
    obs = {
        "main_images": torch.randint(256, (2, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(256, (2, 1, 224, 224, 3), dtype=torch.uint8),
        "states": torch.randn(2, 14),
        "base_actions": torch.randn(2, 10, 7),
        "base_action_mask": torch.ones(2, 10, dtype=torch.bool),
    }
    observed_sizes = []
    handle = policy.encoders[0].register_forward_pre_hook(
        lambda module, args: observed_sizes.append(tuple(args[0].shape[-2:]))
    )
    deterministic, _ = policy.predict_action_batch(obs, mode="eval")
    assert torch.count_nonzero(deterministic) == 0
    actions, infos = policy.predict_action_batch(obs, mode="train")
    handle.remove()
    assert observed_sizes == [(128, 128), (128, 128)]
    assert infos["forward_inputs"]["main_images"].shape[1:3] == (224, 224)
    assert infos["forward_inputs"]["states"].shape == (2, 94)
    out = policy.default_forward(infos["forward_inputs"])
    torch.testing.assert_close(out["logprobs"], infos["prev_logprobs"], rtol=1e-5, atol=1e-5)
    assert torch.count_nonzero(actions[..., 6]) == 0
    entropy = reshape_entropy(out["entropy"], "chunk_level", action_dim=7, batch_size=2)
    mask = reshape_entropy_mask(torch.ones(2, 1, dtype=torch.bool), "chunk_level", 2)
    assert entropy.shape == mask.shape == (2,)
    torch.testing.assert_close(masked_mean(entropy, mask), entropy.mean())
    assert torch.isfinite(out["values"]).all()


def test_cache_clipping_and_gripper():
    cache = BaseActionCache(2, 10, 7, torch.device("cpu"))
    cache.actions.fill_(0.8)
    cache.position.zero_()
    scale = torch.tensor([0.3, 0.3, 0.3, 0.1, 0.1, 0.3, 0.0])
    result = cache.compose(torch.full((2, 7), 2.0), scale)
    torch.testing.assert_close(result[:, 0], torch.ones(2))
    torch.testing.assert_close(result[:, 6], torch.full((2,), 0.8))
    assert torch.equal(cache.position, torch.ones(2, dtype=torch.long))
    cache.invalidate(torch.tensor([0]))
    assert cache.position.tolist() == [10, 1]


def test_smoke_config(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("REPO_PATH", str(root))
    with initialize_config_dir(version_base="1.1", config_dir=str(root / "examples/embodiment/config")):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_slz_smoke")
    validate_residual_cfg(cfg)
    assert "target_kl" not in cfg.algorithm
    assert cfg.env.train.init_params.use_hand_camera
    assert cfg.env.train.init_params.reward_mode == "dense"
    assert cfg.actor.model.encoder_input_size == 128
    assert cfg.runner.max_epochs == 2
