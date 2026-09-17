"""Collect Pour demonstrations on the NUC; --check never imports robot code."""

import argparse
import json
import os
import re
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.pour_water_geometry import PourWaterGeometry
from rlinf.utils.pour_collection import validate_collection_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=2, help="Successful episodes to retain"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", default="realworld_collect_data_pour_spacemouse")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", args.run_name):
        parser.error("Use a simple run name without directory separators.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.config):
        parser.error("Use a config name, not a path.")
    repo = Path(__file__).resolve().parents[2]
    output = repo / "logs/pour_spacemouse" / args.run_name
    os.environ["EMBODIED_PATH"] = str(repo / "examples/embodiment")
    with initialize_config_dir(
        version_base="1.1", config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name=args.config, overrides=args.overrides)
    cfg.runner.num_data_episodes = args.episodes
    cfg.runner.logger.log_path = str(output)
    validate_collection_config(cfg)
    PourWaterGeometry(**OmegaConf.to_container(cfg.pour_task, resolve=True))
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    if args.check:
        print("Configuration OK. No Ray/ROS, camera, SpaceMouse or robot initialized.")
        return
    if output.exists():
        raise FileExistsError(f"Choose a new run name: {output}")
    for device in [
        os.environ.get(
            "RLINF_KEYBOARD_DEVICE",
            "/dev/input/by-id/usb-Cherry_GmbH_CHERRY_Corded_Device-event-kbd",
        ),
        "/dev/input/by-id/usb-3Dconnexion_SpaceMouse_Compact-event-if00",
    ]:
        if not os.access(device, os.R_OK):
            raise RuntimeError(f"Missing/unreadable input device: {device}")
    import pyrealsense2 as rs

    connected = {d.get_info(rs.camera_info.serial_number) for d in rs.context().devices}
    missing = set(cfg.env.eval.override_cfg.camera_serials) - connected
    if missing:
        raise RuntimeError(f"Missing cameras: {missing}; detected: {connected}")
    print(
        "Startup/reset WILL MOVE THE ROBOT. The target is a simulation reference, not a measured hardware pose."
    )
    print(
        "Stop training, verify fixed cup/workspace and keep clear. R=ready; S=success; F=failure; X=abort. No gripper control."
    )
    if (
        input("Type START to initialize the robot (anything else cancels): ").strip()
        != "START"
    ):
        return
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "pour_spacemouse_v1",
                "state": "joint7 + TCP base xyz3 + base Euler xyz3 + fixed gripper -1",
                "images": "main=third-view wrist_2; extra_view[0]=wrist_1, RGB224 padded",
                "actions": "native normalized six-axis SpaceMouse commands before environment scale/clipping",
                "raw": "T actions, T+1 observations; completed successes/failures/aborts retained",
                "lerobot": "successful episodes only; observation t paired with action t; no zero-frame or rotation cleaning",
            },
            indent=2,
        )
    )
    from rlinf.scheduler import Cluster, ComponentPlacement
    from rlinf.workers.env.pour_collection_worker import PourCollectionWorker

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster).get_strategy("env")
    worker = PourCollectionWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=placement
    )
    worker.run().wait()


if __name__ == "__main__":
    main()
