"""Regress seven-dimensional Gaussian residual control without GPU rollouts."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from test_residual_policy import encoder_path as encoder_path
from test_residual_policy import model_config, observation

from rlinf.envs.residual import BaseActionCache
from rlinf.models.embodiment.residual_policy.residual_policy import ResidualPolicy
from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg

ROOT = Path(__file__).resolve().parents[2]
SCALE = torch.tensor([0.4] * 6 + [2.0])


def test_experiment_config_enables_seventh_gaussian_channel():
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_gripper_a4")
    validate_residual_cfg(cfg)
    assert list(cfg.actor.model.residual_action_indices) == list(range(7))
    assert list(cfg.actor.model.residual_action_scale) == [0.4] * 6 + [2.0]
    assert list(cfg.env.train.residual.action_scale) == [0.4] * 6 + [2.0]
    assert list(cfg.env.eval.residual.action_scale) == [0.4] * 6 + [2.0]
    assert list(cfg.actor.model.initial_logstd) == [-0.5] * 6 + [-1.5]
    assert cfg.algorithm.entropy_bonus == 0.0
    assert cfg.runner.resume_dir is None and cfg.runner.ckpt_path is None
    assert dict(cfg.cluster.component_placement) == {"actor,env,rollout": "0,1"}
    assert cfg.env.train.video_cfg.save_video
    assert cfg.env.train.video_cfg.record_rollout_interval == 1
    assert cfg.env.eval.video_cfg.save_video
    cfg.actor.model.residual_action_indices = list(range(6))
    with pytest.raises(ValueError, match="Positive residual scales"):
        validate_residual_cfg(cfg)


def test_seventh_channel_rollout_likelihood_and_ppo_gradient(encoder_path):
    from rlinf.algorithms.losses import compute_decoupled_ppo_actor_critic_loss

    torch.manual_seed(41)
    cfg = model_config(encoder_path, use_state=True)
    cfg.residual_action_indices = list(range(7))
    cfg.initial_logstd = [-0.5] * 6 + [-1.5]
    cfg.logstd_range = [-4.0, 0.0]
    policy = ResidualPolicy(cfg)
    torch.testing.assert_close(
        policy.actor_logstd.detach(), torch.tensor([[-0.5] * 6 + [-1.5]])
    )
    torch.testing.assert_close(
        policy._action_std(policy.actor_logstd).detach(),
        torch.tensor([[math.exp(-0.5)] * 6 + [math.exp(-1.5)]]),
    )
    obs = observation(batch=8, use_state=True)
    baseline, _ = policy.predict_action_batch(obs, mode="eval")
    assert torch.count_nonzero(baseline) == 0
    actions, rollout = policy.predict_action_batch(obs, mode="train")
    output = policy(forward_inputs=rollout["forward_inputs"])
    torch.testing.assert_close(output["logprobs"], rollout["prev_logprobs"])
    torch.testing.assert_close(actions[:, 0], rollout["forward_inputs"]["action"])
    assert torch.count_nonzero(actions[..., 6]) == 8
    assert output["entropy"].shape == (8, 1, 7)
    assert torch.isfinite(output["entropy"]).all()
    torch.testing.assert_close(
        output["logprobs"][:, 6],
        torch.distributions.Normal(0.0, math.exp(-1.5)).log_prob(actions[:, 0, 6]),
    )
    # Reward positive gripper residuals: actual PPO should move its mean positive.
    advantages = actions[:, 0, 6:7].sign()
    loss, _ = compute_decoupled_ppo_actor_critic_loss(
        logprobs=output["logprobs"].sum(-1, keepdim=True),
        old_logprobs=rollout["prev_logprobs"].sum(-1, keepdim=True),
        advantages=advantages,
        values=output["values"],
        prev_values=rollout["prev_values"],
        returns=rollout["prev_values"] + advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        value_loss_coef=0.25,
        huber_delta=10.0,
    )
    loss.backward()
    assert torch.isfinite(policy.actor_mean.bias.grad).all()
    assert policy.actor_mean.bias.grad[6] < 0
    assert torch.isfinite(policy.actor_logstd.grad).all()
    assert policy.actor_logstd.grad[0, 6].abs() > 0
    torch.optim.SGD(policy.parameters(), lr=1e-3).step()
    assert policy.actor_mean.bias[6] > 0


def controller_with_state(open_state):
    """Run the real controller method with only its physics superclass replaced."""
    path = (
        ROOT / "rlinf/envs/maniskill/tasks/digital_twin/controller/safe_pd_joint_pos.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SafePDJointPosMimicController"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_action"
    )

    class PhysicsStub:
        def set_action(self, action):
            self.executed = action.clone()

    wrapper = ast.ClassDef(
        name="Controller",
        bases=[ast.Name(id="PhysicsStub", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = {"PhysicsStub": PhysicsStub, "torch": torch, "Array": torch.Tensor}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    controller = namespace["Controller"]()
    controller.device = torch.device("cpu")
    controller.config = SimpleNamespace(binary_gripper_action=True)
    controller._action_low = torch.tensor([-1.0])
    controller._action_high = torch.tensor([1.0])
    controller._gripper_action_scale = 1.0
    controller._binary_gripper_threshold = 0.5
    controller._use_zero_one_gripper_action = False
    controller._open_command = 1.0
    controller._close_command = -1.0
    controller._gripper_open_state = torch.tensor([[open_state]])
    controller._binary_action = torch.tensor([[1.0 if open_state else -1.0]])
    return controller


@pytest.mark.parametrize(
    "base,residual,initial_open,expected_open",
    [
        (-1.0, 0.0, True, False),  # Baseline closes.
        (-1.0, 0.5, True, True),  # Suppress early close by entering hold band.
        (-1.0, 0.5, False, False),  # Hold is not an unconditional flip.
        (-1.0, 0.75, False, True),  # Reopen at the inclusive +0.5 boundary.
        (1.0, -0.75, True, False),  # Close at the inclusive -0.5 boundary.
        (1.0, -0.5, False, False),
        (1.0, 0.0, False, True),
        (-1.0, 3.0, False, True),  # Raw Gaussian tail is clipped only for execution.
        (-1.2, 1.0, False, True),  # Scale=2 has margin for this out-of-range base.
        (-1.6, 1.0, False, False),  # Larger base magnitude can still prevent reopening.
    ],
)
def test_composition_and_actual_gripper_hysteresis(
    base, residual, initial_open, expected_open
):
    cache = BaseActionCache(1, 1, 7, torch.device("cpu"))
    cache.position.zero_()
    cache.actions[0, 0, 6] = base
    raw = torch.full((1, 7), 0.25)
    raw[0, 6] = residual
    executed = cache.compose(raw, SCALE)
    torch.testing.assert_close(executed[0, :6], torch.full((6,), 0.1))
    assert raw[0, 6].item() == residual
    controller = controller_with_state(initial_open)
    controller.set_action(executed[:, 6:7])
    assert controller._gripper_open_state.item() == expected_open
    assert controller.executed.item() == (1.0 if expected_open else -1.0)


def test_initial_exploration_thresholds():
    sigma = math.exp(-1.5)
    suppress_early_close = 0.5 * math.erfc(0.25 / sigma / math.sqrt(2))
    reopen = 0.5 * math.erfc(0.75 / sigma / math.sqrt(2))
    at_least_one = -math.expm1(120 * math.log1p(-reopen))
    assert 0.13 < suppress_early_close < 0.14
    assert 0.00038 < reopen < 0.00040
    assert 0.045 < at_least_one < 0.046


@pytest.mark.parametrize("initial", [-0.5, [-0.5] * 6 + [-1.5]])
def test_scalar_and_per_dimension_logstd_validation(initial):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_gripper_a4")
    cfg.actor.model.initial_logstd = initial
    validate_residual_cfg(cfg)


@pytest.mark.parametrize(
    "initial",
    [
        [],
        [-0.5] * 6,
        [-0.5] * 8,
        [-0.5] * 6 + [-5.0],
        [-0.5] * 6 + [float("nan")],
        [-0.5] * 6 + [float("inf")],
        float("nan"),
        1.0,
    ],
)
def test_invalid_initial_logstd_rejected(initial):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base=None
    ):
        cfg = compose(config_name="maniskill_pick_and_place_ppo_residual_gripper_a4")
    cfg.actor.model.initial_logstd = initial
    with pytest.raises(ValueError, match="initial_logstd"):
        validate_residual_cfg(cfg)
