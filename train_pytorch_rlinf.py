#!/usr/bin/env python3

"""Run OpenPI PyTorch training with RLinf custom configs registered."""

from __future__ import annotations

import runpy
from pathlib import Path

import openpi.training.config as official_config

from rlinf.models.embodiment.openpi import dataconfig as rlinf_config


for config in rlinf_config._CONFIGS:
    official_config._CONFIGS_DICT[config.name] = config

if hasattr(official_config, "_CONFIGS"):
    configs = official_config._CONFIGS

    existing_names = {
        config.name
        for config in configs
    }

    additions = [
        config
        for config in rlinf_config._CONFIGS
        if config.name not in existing_names
    ]

    if isinstance(configs, tuple):
        official_config._CONFIGS = configs + tuple(additions)
    elif isinstance(configs, list):
        official_config._CONFIGS.extend(additions)

script = (
    Path(__file__).resolve().parent
    / "third_party/openpi/scripts/train_pytorch.py"
)

runpy.run_path(
    str(script),
    run_name="__main__",
)
