# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Check PnP eval configuration and CLI without workers or hardware."""

import ast
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.realworld_eval import evaluation_checkpoint

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("policy", ["cnn_dual_state", "openpi_residual_shared"])
@pytest.mark.parametrize("length", [60, 120, 240])
def test_pnp_eval_contract(policy, length, monkeypatch, tmp_path):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    monkeypatch.setenv("RLINF_LOG_PATH", str(tmp_path))
    monkeypatch.setenv("RLINF_EVAL_RUN_ID", "unit-test")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(ROOT / "examples/embodiment/config")
    ):
        overlay = compose(config_name=f"real_eval_pick_and_place_{policy}")
        assert overlay.evaluation.episode_length == 120
        overlay.evaluation.episode_length = length
        source = compose(config_name=overlay.evaluation.training_config)
        cfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(source)), overlay)
    if policy == "cnn_dual_state":
        checkpoint = tmp_path / "weights.pt"
        checkpoint.touch()
        cfg.runner.ckpt_path = str(checkpoint)
    entry = ROOT / "examples/embodiment/eval_realworld.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"))
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_eval"
    )
    namespace = {"evaluation_checkpoint": evaluation_checkpoint}
    exec(  # noqa: S102 -- Only the hardware-free validator, not the entrypoint.
        compile(ast.Module(body=[validator], type_ignores=[]), str(entry), "exec"),
        namespace,
    )
    namespace["validate_eval"](cfg)
    assert cfg.cluster.num_nodes == 2
    assert cfg.cluster.node_groups[1].node_ranks == 1
    assert cfg.cluster.component_placement.rollout.placement == 0
    assert cfg.cluster.component_placement.env.node_group == "real"
    assert cfg.actor.model.image_num == 2 and cfg.actor.model.state_dim == 14
    assert cfg.env.sim_real_alignment == source.env.sim_real_alignment
    for domain in (cfg.env.train, cfg.env.eval):
        assert domain.env_type == "realworld"
        assert domain.init_params.id == "FrankaPickAndPlaceEnv-v1"
        assert "sim_backend" not in domain.init_params
        assert "co_training_env_cfg" not in domain
        assert domain.max_episode_steps == length
        assert domain.max_steps_per_rollout_epoch == length
        assert not domain.auto_reset and not domain.ignore_terminations
        assert domain.main_image_key == "wrist_2"
        assert domain.keyboard_reward_wrapper == "pnp_human"
        assert domain.human_feedback_cfg.wait_for_reset_ready
        assert domain.video_cfg.include_extra_views and domain.video_cfg.save_video
    OmegaConf.to_container(cfg, resolve=True)


@pytest.fixture
def eval_cli(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is needed for argument parsing tests")
    script = tmp_path / "run_pnp_realworld_eval.sh"
    shutil.copyfile(ROOT / "examples/embodiment" / script.name, script)
    (tmp_path / "run_realworld_eval.sh").write_text(
        'printf "%s\\n" "$@"\n', encoding="utf-8"
    )
    return bash, script


@pytest.mark.parametrize(
    "args",
    [
        ["--episode-length", "0"],
        ["--episode-length", "-1"],
        ["--episode-length", "1.5"],
        ["--episodes", "0"],
        ["--step", "20"],
        ["--initial", "--step", "0"],
        ["--policy", "invalid"],
    ],
)
def test_pnp_cli_rejects_invalid(eval_cli, args):
    result = subprocess.run(
        [*map(str, eval_cli), *args], capture_output=True, text=True
    )
    assert result.returncode == 2


@pytest.mark.parametrize("length", [120, 240])
def test_pnp_cli_initial(eval_cli, length):
    args = [] if length == 120 else ["--episode-length", str(length)]
    result = subprocess.run(
        [*map(str, eval_cli), "--initial", "--check-only", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "runner.ckpt_path=null" in result.stdout
    assert f"evaluation.episode_length={length}" in result.stdout
    assert "evaluation.num_episodes=40" in result.stdout
    assert "evaluation.check_only=true" in result.stdout


def test_pnp_cli_cnn_checkpoint(eval_cli, tmp_path):
    checkpoint = tmp_path / "full_weights.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            *map(str, eval_cli),
            "--policy",
            "cnn",
            "--checkpoint",
            str(checkpoint),
            "--episodes",
            "20",
            "--episode-length",
            "240",
            "--sample",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "real_eval_pick_and_place_cnn_dual_state" in result.stdout
    assert f"runner.ckpt_path={checkpoint}" in result.stdout
    assert "evaluation.num_episodes=20" in result.stdout
    assert "evaluation.episode_length=240" in result.stdout
    assert "evaluation.policy_mode=sample" in result.stdout
