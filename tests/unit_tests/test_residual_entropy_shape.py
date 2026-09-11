"""CPU regression for residual entropy and masked PPO microbatch accumulation.

Extract the actual methods with AST to avoid loading vision/GPU dependencies.
Run: .venv/bin/python tests/unit_tests/test_residual_entropy_shape.py
"""
import ast
from pathlib import Path
from typing import Any, Optional
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]

def load_policy():
    ns = {"torch": torch, "Any": Any, "Optional": Optional}
    tree = ast.parse((ROOT / "rlinf/utils/utils.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("masked_mean", "reshape_entropy", "reshape_entropy_mask"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "utils.py", "exec"), ns)
    class CNNPolicy:
        def default_forward(self, forward_inputs, **kwargs):
            action = forward_inputs["action"]
            dist = torch.distributions.Normal(torch.zeros_like(action), self.logstd.exp().expand_as(action))
            return {"entropy": dist.entropy(), "logprobs": dist.log_prob(action)}
    ns["CNNPolicy"] = CNNPolicy
    tree = ast.parse((ROOT / "rlinf/models/embodiment/residual_policy/residual_policy.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResidualPolicy")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("default_forward", "_expand_actions")]
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "residual_policy.py", "exec"), ns)
    p = ns["ResidualPolicy"]()
    p.action_indices = torch.arange(6)
    p.residual_cfg = type("Config", (), {"action_dim": 7})()
    return p, ns

class EntropyRegression(unittest.TestCase):
    def test_shapes_and_mask(self):
        p, ns = load_policy()
        p.logstd = torch.zeros(6, requires_grad=True)
        for b in (1, 20, 120):
            out = p.default_forward({"action": torch.zeros(b, 7)})
            self.assertEqual(out["logprobs"].shape, (b, 7))
            self.assertEqual(out["entropy"].shape, (b, 1, 7))
            self.assertTrue((out["entropy"][..., 6] == 0).all())
            for kind in ("chunk_level", "action_level"):
                e = ns["reshape_entropy"](out["entropy"], kind, 7, b)
                self.assertEqual(e.shape, (b,) if kind == "chunk_level" else (b, 1))
                for mask in (torch.ones(b, 1, dtype=torch.bool), torch.arange(b).reshape(b, 1) % 2 == 0):
                    mask = ns["reshape_entropy_mask"](mask, kind, b)
                    mean = ns["masked_mean"](e, mask)
                    torch.testing.assert_close(mean, torch.tensor(8.513631))
                z = ns["masked_mean"](e, ns["reshape_entropy_mask"](torch.zeros(b, 1, dtype=torch.bool), kind, b))
                self.assertEqual(z.item(), 0)

    def test_microbatch_gradient(self):
        results = []
        for micro in (20, 120):
            p, ns = load_policy()
            p.logstd = torch.full((6,), -0.2, requires_grad=True)
            total = 0.0
            for start in range(0, 120, micro):
                out = p.default_forward({"action": torch.zeros(micro, 7)})
                e = ns["reshape_entropy"](out["entropy"], "chunk_level", 7, micro)
                mask = ns["reshape_entropy_mask"](torch.ones(micro, 1, dtype=torch.bool), "chunk_level", micro)
                loss = ns["masked_mean"](e, mask) / (120 // micro)
                total += loss.item()
                loss.backward()
            results.append((total, p.logstd.grad))
        self.assertAlmostEqual(results[0][0], results[1][0], places=5)
        torch.testing.assert_close(results[0][1], results[1][1])
        torch.testing.assert_close(results[1][1], torch.ones(6))

if __name__ == "__main__":
    unittest.main()
