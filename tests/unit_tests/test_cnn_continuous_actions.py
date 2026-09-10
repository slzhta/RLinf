"""Exercise the actual CNN distribution methods without a backbone or simulator."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from torch.distributions import Bernoulli, Normal


def policy_type():
    source = Path(__file__).resolve().parents[2] / (
        "rlinf/models/embodiment/cnn_policy/cnn_policy.py"
    )
    cls = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.ClassDef) and node.name == "CNNPolicy"
    )
    names = {
        "_action_std",
        "_continuous_action_statistics",
        "_hybrid_action_statistics",
        "_generate_actions",
        "default_forward",
    }
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    scope = {
        "torch": torch,
        "Normal": Normal,
        "Bernoulli": Bernoulli,
        "Optional": Optional,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    return type("DistributionPolicy", (), {name: scope[name] for name in names})


class ContinuousActionTests(unittest.TestCase):
    def make_policy(self, scaled=False, dtype=torch.float32):
        policy = policy_type()()
        policy.cfg = SimpleNamespace(
            num_action_chunks=1,
            action_dim=6,
            logstd_range=[-4.0, 0.0],
            action_std_scale=[1.0] * 6,
            std_range=None,
            binary_action_temperature=1.0,
        )
        policy._binary_action_indices = ()
        policy._continuous_action_indices = tuple(range(6))
        policy.action_scale = torch.tensor(2.0) if scaled else None
        policy.action_bias = torch.tensor(0.5) if scaled else None
        mean = torch.tensor(
            [[0.0, 0.3, -0.5, 1.2, -3.0, 9.0]], dtype=dtype, requires_grad=True
        )
        logstd = torch.full_like(mean, -1.0, requires_grad=True)
        policy._actor_forward_from_processed_tensors = lambda *a, **kw: (
            mean,
            mean,
            mean,
            logstd,
        )
        policy.preprocess_env_obs = lambda obs: obs
        return policy, mean, logstd

    def check_parity(self, policy, mode):
        action, chunks, old_logprobs, _, _ = policy._generate_actions(
            None, torch.zeros(1), None, False, mode
        )
        result = policy.default_forward(
            {"main_images": torch.zeros(1), "action": action.detach()},
            compute_values=False,
        )
        self.assertEqual(tuple(chunks.shape), (1, 1, 6))
        self.assertTrue(torch.isfinite(old_logprobs).all())
        torch.testing.assert_close(result["logprobs"], old_logprobs)
        torch.testing.assert_close(
            torch.exp(result["logprobs"] - old_logprobs), torch.ones_like(old_logprobs)
        )
        return action, result

    def test_eval_tanh_and_ppo_parity(self):
        for scaled in (False, True):
            policy, mean, _ = self.make_policy(scaled)
            action, _ = self.check_parity(policy, "eval")
            expected = torch.tanh(mean)
            if scaled:
                expected = expected * 2.0 + 0.5
            torch.testing.assert_close(action, expected)

    def test_sampled_bounds_and_finite_ppo_gradients(self):
        torch.manual_seed(12)
        policy, mean, logstd = self.make_policy()
        action, result = self.check_parity(policy, "train")
        self.assertTrue((action.abs() <= 1).all())
        (-result["logprobs"].sum()).backward()
        self.assertTrue(torch.isfinite(mean.grad).all())
        self.assertTrue(torch.isfinite(logstd.grad).all())

    def test_reduced_precision_saturation_remains_finite(self):
        policy, _, _ = self.make_policy(dtype=torch.bfloat16)
        self.check_parity(policy, "eval")

    def test_density_includes_inverse_and_jacobian(self):
        policy, mean, logstd = self.make_policy(scaled=True)
        bounded = torch.full_like(mean, 0.2)
        logprob, _ = policy._continuous_action_statistics(
            mean, logstd.exp(), bounded * 2 + 0.5
        )
        expected = Normal(mean, logstd.exp()).log_prob(
            torch.atanh(bounded)
        ) - torch.log(2 * (1 - bounded.square()) + 1e-6)
        torch.testing.assert_close(logprob, expected)

    def test_hybrid_branch_keeps_binary_gripper(self):
        policy, mean, _ = self.make_policy()
        policy._binary_action_indices = (5,)
        policy._continuous_action_indices = tuple(range(5))
        action, _ = self.check_parity(policy, "eval")
        torch.testing.assert_close(action[:, :5], torch.tanh(mean[:, :5]))
        self.assertEqual(float(action[0, 5]), 1.0)


if __name__ == "__main__":
    unittest.main()
