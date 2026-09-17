"""CPU-only collection contracts; never import the real robot environment."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from rlinf.utils.pour_collection import (
    export_success,
    validate_collection_config,
    validate_observation,
)
from rlinf.utils.realworld_collection import EpisodeCapture

ROOT = Path(__file__).resolve().parents[2]


def config():
    base = OmegaConf.load(
        ROOT / "examples/embodiment/config/env/realworld_pour_water_co_rl.yaml"
    )
    cfg = OmegaConf.load(
        ROOT / "examples/embodiment/config/realworld_collect_data_pour_spacemouse.yaml"
    )
    cfg.env.eval = OmegaConf.merge(base, cfg.env.eval)
    return cfg


def test_config_and_simulation_reference():
    cfg = config()
    validate_collection_config(cfg)
    sim = OmegaConf.load(
        ROOT
        / "examples/embodiment/config/sim_rl_pour_bc3000_dual_state_n32_ue2_rl500.yaml"
    )
    assert cfg.pour_task == sim.pour_task
    assert cfg.env.eval.auto_reset is False
    for key, value in [
        ("no_gripper", True),
        ("spacemouse_gripper_enabled", True),
        ("auto_reset", True),
    ]:
        broken = copy.deepcopy(cfg)
        broken.env.eval[key] = value
        with pytest.raises(ValueError):
            validate_collection_config(broken)


def test_six_axis_spacemouse_wiring_keeps_existing_default():
    # Exercise the actual factory method with wrappers replaced by inert objects.
    source = ast.parse((ROOT / "rlinf/envs/realworld/realworld_env.py").read_text())
    cls = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "RealWorldEnv"
    )
    method = copy.deepcopy(
        next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_create_env"
        )
    )
    observed = []
    native = SimpleNamespace(config=SimpleNamespace(is_dummy=False))
    namespace = dict(
        copy=copy,
        gym=SimpleNamespace(make=lambda **_: native),
        SpacemouseIntervention=lambda env, gripper_enabled: (
            observed.append(gripper_enabled) or env
        ),
        HumanPnPRewardDoneWrapper=lambda env, **_: env,
        Quat2EulerWrapper=lambda env: env,
        OmegaConf=OmegaConf,
    )
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), "factory", "exec"),
        namespace,
    )
    cfg = config().env.eval
    obj = SimpleNamespace(cfg=cfg, override_cfg={}, worker_info=None)
    namespace["_create_env"](obj, 0)
    assert observed == [False]
    del cfg.spacemouse_gripper_enabled
    namespace["_create_env"](obj, 0)
    assert observed == [False, True]


def test_capture_alignment_dualview_language_and_rotations(tmp_path):
    obs = dict(
        main_images=np.full((1, 224, 224, 3), 10, np.uint8),
        extra_view_images=np.full((1, 1, 224, 224, 3), 20, np.uint8),
        states=np.zeros((1, 14), np.float32),
        task_descriptions=["Pour the light green ball into the purple cup"],
    )
    obs["states"][0, -1] = -1
    validate_observation(obs)
    capture = EpisodeCapture()
    capture.observation(obs)
    obs["states"][0, 0] = 0.25
    action = np.array([[0.1, 0.2, 0.3, -0.4, 0.5, -0.6]], np.float32)
    capture.transition(action, [1], [True], [False], obs, 1.0)
    record = dict(completed=True, steps=1, success=True)
    capture.save(tmp_path / "episode", record, {})
    with np.load(tmp_path / "episode/trajectory.npz", allow_pickle=False) as data:
        arrays = dict(data)
    written = []
    writer = SimpleNamespace(
        add_episode=lambda **k: written.append(k), finalize=lambda: None
    )
    export_success(writer, arrays)
    row = written[0]
    assert row["states"][0, 0] == 0
    assert arrays["obs_states"][1, 0] == 0.25
    assert row["images"][0, 0, 0, 0] == 10
    assert row["wrist_images"][0, 0, 0, 0] == 20
    np.testing.assert_array_equal(row["actions"], action)
    assert row["task"] == obs["task_descriptions"][0]
    assert row["dones"].tolist() == [True]


def test_bad_observation_rejected():
    with pytest.raises(ValueError):
        validate_observation(
            dict(
                main_images=np.zeros((1, 128, 128, 3), np.uint8),
                extra_view_images=np.zeros((1, 1, 224, 224, 3), np.uint8),
                states=np.zeros((1, 14)),
                task_descriptions=["Pour"],
            )
        )


def test_lerobot_roundtrip(tmp_path):
    import json

    pq = pytest.importorskip("pyarrow.parquet")
    from rlinf.data.lerobot_writer import LeRobotDatasetWriter

    writer = LeRobotDatasetWriter(
        str(tmp_path / "lerobot"),
        fps=10,
        image_shape=(224, 224, 3),
        state_dim=14,
        action_dim=6,
        has_wrist_image=True,
        has_extra_view_image=False,
        use_incremental_stats=True,
        stats_sample_ratio=1.0,
    )
    export_success(
        writer,
        {
            "actions": np.array([[0, 0, 0, 0.4, -0.6, 0]], np.float32),
            "obs_main_images": np.zeros((2, 224, 224, 3), np.uint8),
            "obs_extra_view_images": np.full((2, 1, 224, 224, 3), 99, np.uint8),
            "obs_states": np.tile(np.r_[np.zeros(13), -1], (2, 1)),
            "obs_task_descriptions": np.array(["Pour the ball"] * 2),
            "terminated": np.array([True]),
            "truncated": np.array([False]),
        },
    )
    root = tmp_path / "lerobot"
    row = pq.read_table(next(root.glob("data/*/*.parquet"))).to_pylist()[0]
    assert len(row["state"]) == 14 and len(row["actions"]) == 6
    assert row["actions"][3] == pytest.approx(0.4)
    assert row["actions"][4] == pytest.approx(-0.6)
    assert row["image"]["bytes"] and row["wrist_image"]["bytes"]
    task = json.loads((root / "meta/tasks.jsonl").read_text().strip())
    assert task == {"task_index": row["task_index"], "task": "Pour the ball"}
