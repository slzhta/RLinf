# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate real-only residual modes without allocating workers or hardware."""

import ast
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.residual_policy.validation import validate_residual_cfg
from rlinf.utils.realworld_eval import evaluation_checkpoint

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=[False, True], ids=["training", "standalone_eval"])
def real_cfg(request):
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/embodiment/config")
    ):
        cfg = compose(
            config_name="real_rl_peg_insertion_openpi_residual_wrist_state_t240_ue4"
        )
        if request.param:
            overrides = compose(
                config_name="real_eval_peg_insertion_openpi_residual_wrist_state"
            )
            cfg = OmegaConf.merge(
                OmegaConf.create(OmegaConf.to_container(cfg)), overrides
            )
        return cfg


def test_real_modes_validate(real_cfg):
    validate_residual_cfg(real_cfg)
    assert real_cfg.cluster.num_nodes == 2
    assert real_cfg.env.eval.auto_reset == (not real_cfg.runner.only_eval)


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("cluster.num_nodes", 3, "2 nodes"),
        ("rollout.pipeline_stage_num", 2, "one stage"),
        ("runner.val_check_interval", 5, "periodic eval is disabled"),
        ("env.train.include_states_in_obs", False, "states"),
        ("env.train.ignore_terminations", True, "terminations"),
    ],
)
def test_real_modes_reject_invalid_contract(real_cfg, key, value, match):
    OmegaConf.update(real_cfg, key, value, force_add=True)
    with pytest.raises(ValueError, match=match):
        validate_residual_cfg(real_cfg)


def test_real_modes_enforce_reset_owner(real_cfg):
    real_cfg.env.train.auto_reset = real_cfg.runner.only_eval
    with pytest.raises(ValueError, match="auto_reset=false for standalone eval"):
        validate_residual_cfg(real_cfg)


def test_only_eval_requires_standalone_entry(real_cfg):
    real_cfg.runner.only_eval = True
    OmegaConf.update(real_cfg, "evaluation.num_episodes", 0, force_add=True)
    with pytest.raises(ValueError, match="standalone real-world evaluation"):
        validate_residual_cfg(real_cfg)


def test_check_only_covers_residual_validator(real_cfg):
    if not real_cfg.runner.only_eval:
        return
    source = ROOT / "examples/embodiment/eval_realworld.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_eval"
    )
    namespace = {"evaluation_checkpoint": evaluation_checkpoint}
    exec(  # noqa: S102 -- Execute only the repository's hardware-free validator.
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    namespace["validate_eval"](real_cfg)
    real_cfg.cluster.num_nodes = 3
    with pytest.raises(ValueError, match="2 nodes"):
        namespace["validate_eval"](real_cfg)
