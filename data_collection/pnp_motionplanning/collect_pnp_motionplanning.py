#!/usr/bin/env python3
"""Collect successful PickAndPlaceDigitalTwin-v1 demonstrations with MPlib 0.1.1.

The saved action is exactly the normalized 7-D command passed to env.step().
This script intentionally writes a dependency-free raw NPZ dataset; convert it
to the LeRobot version pinned by RLinf in a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mplib
import numpy as np
import torch
import cv2

# Import registers PickAndPlaceDigitalTwin-v1.
import rlinf.envs.maniskill.tasks.digital_twin  # noqa: F401


TASK = "pick up the purple cube from the taped tray and place it in the plain tray"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
MOVE_GROUP = "panda_hand_tcp"


@dataclass
class CollectorConfig:
    episodes: int = 100
    max_attempts: int = 500
    seed: int = 0
    control_freq: int = 5
    max_episode_steps: int = 120
    position_scale: float = 0.01
    rotation_scale: float = 0.05
    pregrasp_height: float = 0.10
    grasp_tcp_offset: float = 0.011
    lift_height: float = 0.13
    place_tcp_offset: float = 0.013
    retreat_height: float = 0.10
    grasp_settle_steps: int = 8
    grasp_dwell: int = 6
    place_settle_steps: int = 6
    release_dwell: int = 4
    planning_time: float = 3.0


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    sin_half = np.linalg.norm(q[1:])
    if sin_half < 1e-8:
        return 2.0 * q[1:]
    angle = 2.0 * math.atan2(sin_half, np.clip(q[0], -1.0, 1.0))
    return q[1:] / sin_half * angle


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / np.linalg.norm(q)
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    return (math.sin((1 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / math.sin(theta)


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose wxyz poses [x,y,z,qw,qx,qy,qz]."""
    ap, aq, bp, bq = a[:3], a[3:], b[:3], b[3:]
    # Rotate bp by aq via q * [0,p] * q^-1.
    rp = quat_mul(quat_mul(aq, np.r_[0.0, bp]), quat_inv(aq))[1:]
    return np.r_[ap + rp, quat_mul(aq, bq)]


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    qi = quat_inv(pose[3:])
    p = quat_mul(quat_mul(qi, np.r_[0.0, -pose[:3]]), quat_inv(qi))[1:]
    return np.r_[p, qi]


class RawEpisode:
    def __init__(self) -> None:
        self.base_images: list[np.ndarray] = []
        self.wrist_images: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def append_observation(self, info: dict, action: np.ndarray) -> None:
        extracted = info["extracted_obs"]
        self.base_images.append(to_numpy(extracted["main_images"])[0].astype(np.uint8))
        wrist = to_numpy(extracted["extra_view_images"])[0]
        self.wrist_images.append(wrist[0].astype(np.uint8))
        self.states.append(to_numpy(extracted["states"])[0].astype(np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))

    def append_result(self, reward: Any, terminated: Any, truncated: Any) -> None:
        self.rewards.append(float(to_numpy(reward).reshape(-1)[0]))
        self.terminated.append(bool(to_numpy(terminated).reshape(-1)[0]))
        self.truncated.append(bool(to_numpy(truncated).reshape(-1)[0]))

    def save(self, path: Path, seed: int) -> None:
        np.savez_compressed(
            path,
            base_images=np.stack(self.base_images),
            wrist_images=np.stack(self.wrist_images),
            states=np.stack(self.states),
            actions=np.stack(self.actions),
            rewards=np.asarray(self.rewards, dtype=np.float32),
            terminated=np.asarray(self.terminated, dtype=np.bool_),
            truncated=np.asarray(self.truncated, dtype=np.bool_),
            task=np.asarray(TASK),
            seed=np.asarray(seed, dtype=np.int64),
        )

    def save_videos(self, directory: Path, stem: str, fps: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, frames in (("base", self.base_images), ("wrist", self.wrist_images)):
            if not frames:
                continue
            height, width = frames[0].shape[:2]
            output = directory / f"{stem}_{name}.mp4"
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to create video: {output}")
            try:
                for rgb in frames:
                    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()


def make_mplib_urdf(source_urdf: Path, output_dir: Path) -> tuple[Path, Path]:
    """Create an MPlib URDF with absolute, existing mesh paths.

    Some ManiSkill URDFs refer to generated names such as foo.stl.convex.stl.
    MPlib 0.1.1 does not generate those files, so fall back to foo.stl when it
    exists. The original URDF is never modified.
    """
    source_urdf = source_urdf.resolve()
    tree = ET.parse(source_urdf)
    missing: list[str] = []
    output_dir = output_dir.resolve()
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            # This project keeps franka_description beside panda_umi.urdf.
            relative = filename[len("package://"):]
            if relative.startswith("franka_description/"):
                path = source_urdf.parent / relative
            else:
                path = source_urdf.parent / relative
        else:
            path = Path(filename)
            if not path.is_absolute():
                path = source_urdf.parent / path
        path = path.resolve()
        candidates = [path]
        text = str(path)
        if text.endswith(".convex.stl"):
            candidates.append(Path(text[: -len(".convex.stl")]))
        if text.endswith(".stl.convex.stl"):
            candidates.append(Path(text[: -len(".convex.stl")]))
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        if existing is None:
            missing.append(f"{filename} -> tried: {', '.join(map(str, candidates))}")
        else:
            # MPlib 0.1.1 incorrectly treats an absolute mesh filename as
            # relative to the generated URDF.  Always write a path relative
            # to planner_assets instead.
            if existing.suffix.lower() == ".stl":
                # Some 0.1.1 wheels append ".convex.stl" unconditionally even
                # when Planner(use_convex=False) is requested. Keep compatible
                # aliases inside planner_assets, never beside the source mesh.
                digest = hashlib.sha1(str(existing).encode("utf-8")).hexdigest()[:10]
                mesh_dir = output_dir / "collision_meshes"
                mesh_dir.mkdir(parents=True, exist_ok=True)
                local_mesh = mesh_dir / f"{digest}_{existing.name}"
                convex_alias = Path(str(local_mesh) + ".convex.stl")
                shutil.copy2(existing, local_mesh)
                shutil.copy2(existing, convex_alias)
                mesh.set("filename", str(Path(os.path.relpath(local_mesh, output_dir))))
            else:
                mesh.set("filename", str(Path(os.path.relpath(existing, output_dir))))
    if missing:
        raise FileNotFoundError("MPlib URDF mesh files are missing:\n" + "\n".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "panda_umi_mplib.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)

    # MPlib 0.1.1 crashes in its no-SRDF automatic collision-pair path because
    # link_name_2_idx is accessed before initialization. Generate a minimal
    # SRDF that disables collision checks for directly connected link pairs.
    urdf_root = tree.getroot()
    srdf_root = ET.Element("robot", {"name": urdf_root.get("name", "panda_umi")})
    seen_pairs: set[tuple[str, str]] = set()
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        link1, link2 = parent.get("link"), child.get("link")
        if not link1 or not link2:
            continue
        pair = tuple(sorted((link1, link2)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        ET.SubElement(
            srdf_root,
            "disable_collisions",
            {"link1": link1, "link2": link2, "reason": "Adjacent"},
        )
    generated_srdf = output_dir / "panda_umi_mplib.srdf"
    ET.ElementTree(srdf_root).write(
        generated_srdf, encoding="utf-8", xml_declaration=True
    )
    return output, generated_srdf


def add_srdf_collision_pairs(srdf_path: Path, pairs: set[tuple[str, str]]) -> int:
    """Append disabled collision pairs, returning the number newly added."""
    tree = ET.parse(srdf_path)
    root = tree.getroot()
    existing = {
        tuple(sorted((item.get("link1", ""), item.get("link2", ""))))
        for item in root.findall("disable_collisions")
    }
    added = 0
    for link1, link2 in sorted(pairs):
        pair = tuple(sorted((link1, link2)))
        if not link1 or not link2 or link1 == link2 or pair in existing:
            continue
        ET.SubElement(
            root,
            "disable_collisions",
            {"link1": link1, "link2": link2, "reason": "DefaultPose"},
        )
        existing.add(pair)
        added += 1
    if added:
        tree.write(srdf_path, encoding="utf-8", xml_declaration=True)
    return added


class PnPMotionCollector:
    def __init__(self, args: argparse.Namespace, cfg: CollectorConfig):
        self.args, self.cfg = args, cfg
        target_ee_pose = np.array(
            [
                0.54515648,
                0.03448609,
                -0.01847301,
                3.07488508,
                -0.01941186,
                -0.04468620,
            ],
            dtype=np.float64,
        )
        ee_pose_limit_min = np.array(
            [
                target_ee_pose[0] - 0.15,
                target_ee_pose[1] - 0.15,
                target_ee_pose[2] - 0.025,
                target_ee_pose[3] - 0.01,
                target_ee_pose[4] - 0.01,
                target_ee_pose[5] - 0.50,
            ]
        )
        ee_pose_limit_max = np.array(
            [
                target_ee_pose[0] + 0.15,
                target_ee_pose[1] + 0.15,
                target_ee_pose[2] + 0.15,
                target_ee_pose[3] + 0.01,
                target_ee_pose[4] + 0.01,
                target_ee_pose[5] + 0.50,
            ]
        )
        self.env = gym.make(
            "PickAndPlaceDigitalTwin-v1",
            num_envs=1,
            obs_mode="rgb+segmentation",
            control_mode="pd_ee_body_target_delta_pose_real",
            sim_backend=args.sim_backend,
            reward_mode="dense",
            max_episode_steps=cfg.max_episode_steps,
            sim_config={"sim_freq": 500, "control_freq": cfg.control_freq},
            controller_alignment={
                "use_target_controller": True,
                "use_zero_one_gripper_action": False,
                "target_ee_pose": target_ee_pose.tolist(),
                "ee_pose_limit_min": ee_pose_limit_min.tolist(),
                "ee_pose_limit_max": ee_pose_limit_max.tolist(),
                "action_scale": [cfg.position_scale, cfg.rotation_scale, 1.0],
                "binary_gripper_action": True,
                "binary_gripper_threshold": 0.5,
                "open_command": 1.0,
                "close_command": -1.0,
            },
            task_alignment={
                "object_random_in_source_tray": True,
                "object_random_xy_range": 0.01,
                "object_random_yaw_range": math.pi,
                "use_dense_reward": False,
            },
        )
        self.u = self.env.unwrapped
        robot = self.u.agent.robot
        self.link_names = [link.name for link in robot.links]
        self.joint_names = [joint.name for joint in robot.active_joints]
        planner_urdf, generated_srdf = make_mplib_urdf(
            Path(self.u.agent.urdf_path), args.output_dir / "planner_assets"
        )
        kwargs = dict(
            urdf=str(planner_urdf),
            move_group=MOVE_GROUP,
            # MPlib 0.1.1 defaults to appending ".convex.stl" to every
            # collision mesh.  This URDF already supplies collision STL files.
            use_convex=False,
            user_link_names=self.link_names,
            user_joint_names=self.joint_names,
            joint_vel_limits=np.full(7, 0.35),
            joint_acc_limits=np.full(7, 0.7),
        )
        # Never mutate a user-provided SRDF. Copy it into planner_assets first.
        planner_srdf = generated_srdf
        if args.srdf:
            planner_srdf = args.output_dir / "planner_assets" / "user_mplib.srdf"
            shutil.copy2(Path(args.srdf), planner_srdf)
        kwargs["srdf"] = str(planner_srdf)

        # A hand-written robot often contains fixed gripper/camera meshes that
        # touch in its valid default pose. Discover only the pairs colliding in
        # the simulator's current legal qpos and add them to the generated SRDF.
        initial_qpos = to_numpy(robot.get_qpos())[0].astype(np.float64)
        self.planner = None
        for repair_round in range(8):
            self.planner = mplib.Planner(**kwargs)
            collisions = self.planner.check_for_self_collision(
                self.planner.robot, initial_qpos
            )
            if not collisions:
                break
            pairs = {
                (getattr(item, "link_name1", ""), getattr(item, "link_name2", ""))
                for item in collisions
            }
            added = add_srdf_collision_pairs(planner_srdf, pairs)
            print(
                f"MPlib SRDF repair round {repair_round}: "
                f"collisions={sorted(pairs)}, added={added}",
                flush=True,
            )
            if added == 0:
                raise RuntimeError(
                    "MPlib reports self-collision but exposes no new link pair: "
                    + str([str(item) for item in collisions])
                )
        else:
            raise RuntimeError("failed to repair default-pose SRDF after 8 rounds")
        self.pinocchio = self.planner.robot.get_pinocchio_model()
        self.ee_link_index = self.link_names.index(MOVE_GROUP)
        self.obs = self.info = None
        self.episode = RawEpisode()
        self.last_step_success = False

    def reset(self, seed: int) -> None:
        self.obs, self.info = self.env.reset(seed=seed)
        self.episode = RawEpisode()
        self.last_step_success = False

    def current_qpos(self) -> np.ndarray:
        return to_numpy(self.u.agent.robot.get_qpos())[0].astype(np.float64)

    def fk(self, arm_qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        full = reference_qpos.copy()
        full[:7] = arm_qpos[:7]
        self.pinocchio.compute_forward_kinematics(full)
        pose = self.pinocchio.get_link_pose(self.ee_link_index)
        if isinstance(pose, np.ndarray):
            pose_array = np.asarray(pose, dtype=np.float64).reshape(-1)
            if pose_array.size != 7:
                raise ValueError(
                    f"MPlib get_link_pose returned shape {pose_array.shape}, expected 7 values"
                )
            return pose_array
        return np.r_[np.asarray(pose.p), np.asarray(pose.q)].astype(np.float64)

    def step(self, action: np.ndarray) -> bool:
        self.episode.append_observation(self.info, action)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action[None])
        self.episode.append_result(reward, terminated, truncated)
        done = bool(
            to_numpy(terminated).reshape(-1)[0]
            or to_numpy(truncated).reshape(-1)[0]
        )
        if done:
            success_value = self.info.get("success") if isinstance(self.info, dict) else None
            if success_value is None:
                success_value = self.u.evaluate()["success"]
            self.last_step_success = bool(to_numpy(success_value).reshape(-1)[0])
        else:
            self.last_step_success = False
        return done

    def execute_pose_increment(self, p0: np.ndarray, p1: np.ndarray, gripper: float) -> bool:
        dp = p1[:3] - p0[:3]
        drot = quat_to_rotvec(quat_mul(quat_inv(p0[3:]), p1[3:]))
        pieces = max(
            1,
            int(math.ceil(np.max(np.abs(dp)) / self.cfg.position_scale)),
            int(math.ceil(np.max(np.abs(drot)) / self.cfg.rotation_scale)),
        )
        prev = p0
        for i in range(1, pieces + 1):
            alpha = i / pieces
            target = np.r_[p0[:3] + alpha * dp, slerp(p0[3:], p1[3:], alpha)]
            local_dp = target[:3] - prev[:3]
            local_dr = quat_to_rotvec(quat_mul(quat_inv(prev[3:]), target[3:]))
            action = np.r_[
                np.clip(local_dp / self.cfg.position_scale, -1, 1),
                np.clip(local_dr / self.cfg.rotation_scale, -1, 1),
                gripper,
            ].astype(np.float32)
            if self.step(action):
                return False
            prev = target
        return True

    def plan_and_execute(self, goal_env_base: np.ndarray, gripper: float) -> bool:
        start = self.current_qpos()
        start_mplib = self.fk(start[:7], start)
        start_env = to_numpy(self.u.agent.ee_pose_at_robot_base.raw_pose)[0].astype(
            np.float64
        )
        # panda_hand_tcp has a different fixed-link convention in MPlib and
        # SAPIEN. Calibrate the constant transform at the current qpos:
        #   T_base_mplib = T_base_env * T_env_mplib
        env_to_mplib = compose_pose(inverse_pose(start_env), start_mplib)
        mplib_to_env = inverse_pose(env_to_mplib)
        goal_mplib = compose_pose(goal_env_base, env_to_mplib)
        result = self.planner.plan_qpos_to_pose(
            goal_mplib.tolist(), start, time_step=1.0 / self.cfg.control_freq,
            planning_time=self.cfg.planning_time, wrt_world=True,
            planner_name="RRTConnect", use_point_cloud=False, use_attach=False,
        )
        if result.get("status") != "Success":
            diagnostic_parts = []
            try:
                ik_status, ik_solutions = self.planner.IK(
                    goal_mplib.tolist(), start, n_init_qpos=100, threshold=0.002
                )
                diagnostic_parts.append(
                    f"direct_ik={ik_status}, solutions_shape={np.asarray(ik_solutions).shape}"
                )
            except Exception as exc:  # diagnostic must not hide the original failure
                diagnostic_parts.append(f"direct_ik_error={type(exc).__name__}: {exc}")
            try:
                collisions = self.planner.check_for_self_collision(
                    self.planner.robot, start
                )
                diagnostic_parts.append(
                    "start_self_collisions=" + str([str(item) for item in collisions])
                )
            except Exception as exc:
                diagnostic_parts.append(
                    f"collision_check_error={type(exc).__name__}: {exc}"
                )
            raise RuntimeError(
                f"planning failed: {result.get('status')}; "
                f"start_mplib={np.round(start_mplib, 5).tolist()}; "
                f"start_env={np.round(start_env, 5).tolist()}; "
                f"goal_env={np.round(goal_env_base, 5).tolist()}; "
                f"goal_mplib={np.round(goal_mplib, 5).tolist()}; "
                + "; ".join(diagnostic_parts)
            )
        path = np.asarray(result["position"])
        mplib_poses = [self.fk(q, start) for q in path]
        env_poses = [compose_pose(pose, mplib_to_env) for pose in mplib_poses]
        # Anchor the command path exactly at the controller's observed TCP pose.
        if env_poses:
            env_poses[0] = start_env
            env_poses[-1] = goal_env_base
        for a, b in zip(env_poses[:-1], env_poses[1:]):
            if not self.execute_pose_increment(a, b, gripper):
                return False
        return True

    def hold(self, gripper: float, steps: int) -> bool:
        action = np.zeros(7, dtype=np.float32)
        action[6] = gripper
        for _ in range(steps):
            if self.step(action):
                return False
        return True

    def collect_one(self, seed: int) -> tuple[bool, str]:
        self.reset(seed)
        base_world = to_numpy(self.u.agent.robot.pose.raw_pose)[0].astype(np.float64)
        world_to_base = inverse_pose(base_world)
        cube_world = to_numpy(self.u.cube.pose.raw_pose)[0].astype(np.float64)
        goal_world_p = to_numpy(self.u._get_goal_cube_position())[0].astype(np.float64)
        # Environment actions control SAPIEN's TCP convention. MPlib's TCP
        # convention is calibrated separately inside plan_and_execute().
        tcp_base = to_numpy(self.u.agent.ee_pose_at_robot_base.raw_pose)[0].astype(
            np.float64
        )
        fixed_q = tcp_base[3:]

        def target(world_p: np.ndarray) -> np.ndarray:
            return compose_pose(world_to_base, np.r_[world_p, fixed_q])

        cube_p = cube_world[:3]
        waypoints = [
            (target(cube_p + [0, 0, self.cfg.pregrasp_height]), +1.0),
            (target(cube_p + [0, 0, self.cfg.grasp_tcp_offset]), +1.0),
        ]
        try:
            for pose, grip in waypoints:
                if not self.plan_and_execute(pose, grip):
                    return False, "ended_before_grasp"
            # The target controller accumulates the commanded target pose, but
            # the physical TCP lags behind it. Keep the target unchanged and
            # gripper open until the arm settles before closing.
            if not self.hold(+1.0, self.cfg.grasp_settle_steps):
                return False, "ended_while_settling_at_grasp"
            if not self.hold(-1.0, self.cfg.grasp_dwell):
                return False, "ended_during_grasp"
            grasp_eval = self.u.evaluate()
            is_grasped = bool(to_numpy(grasp_eval["is_grasped"]).reshape(-1)[0])
            if not is_grasped:
                tcp_world = to_numpy(self.u.agent.tcp.pose.p)[0].astype(np.float64)
                cube_now = to_numpy(self.u.cube.pose.p)[0].astype(np.float64)
                return False, (
                    "grasp_failed; "
                    f"tcp_world={np.round(tcp_world, 5).tolist()}; "
                    f"cube_world={np.round(cube_now, 5).tolist()}; "
                    f"tcp_minus_cube={np.round(tcp_world - cube_now, 5).tolist()}"
                )
            # Use the live cube position after grasping.
            live_cube = to_numpy(self.u.cube.pose.p)[0]
            lift = target(live_cube + [0, 0, self.cfg.lift_height])
            preplace = target(goal_world_p + [0, 0, self.cfg.lift_height])
            place = target(goal_world_p + [0, 0, self.cfg.place_tcp_offset])
            for pose in (lift, preplace, place):
                if not self.plan_and_execute(pose, -1.0):
                    return False, "ended_during_transport"
            # Likewise, wait for the object/TCP to settle at the place target
            # before opening the gripper.
            if not self.hold(-1.0, self.cfg.place_settle_steps):
                return False, "ended_while_settling_at_place"
            release_finished = self.hold(+1.0, self.cfg.release_dwell)
            if self.last_step_success:
                return True, "success"
            if not release_finished:
                return False, "ended_during_release"
            retreat = target(goal_world_p + [0, 0, self.cfg.retreat_height])
            self.plan_and_execute(retreat, +1.0)
        except RuntimeError as exc:
            return False, str(exc)

        evaluation = self.u.evaluate()
        success = bool(to_numpy(evaluation["success"]).reshape(-1)[0])
        return success, "success" if success else "task_not_successful"

    def close(self) -> None:
        self.env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-attempts", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-backend", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--srdf", type=str, default="")
    p.add_argument(
        "--save-failure-videos", action="store_true",
        help="also save videos for failed attempts (useful during the first test)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CollectorConfig(episodes=args.episodes, max_attempts=args.max_attempts, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(
        json.dumps({**asdict(cfg), "task": TASK, "state_dim": 14, "action_dim": 7,
                    "cameras": ["3rdview_camera", "hand_camera"]}, indent=2),
        encoding="utf-8",
    )
    log_path = args.output_dir / "collection_log.jsonl"
    collector = PnPMotionCollector(args, cfg)
    saved = 0
    try:
        for attempt in range(cfg.max_attempts):
            if saved >= cfg.episodes:
                break
            seed = cfg.seed + attempt
            success, reason = collector.collect_one(seed)
            record = {"attempt": attempt, "seed": seed, "success": success,
                      "reason": reason, "steps": len(collector.episode.actions)}
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(record, flush=True)
            if success:
                path = args.output_dir / f"episode_{saved:06d}.npz"
                collector.episode.save(path, seed)
                collector.episode.save_videos(
                    args.output_dir / "videos", f"episode_{saved:06d}", cfg.control_freq
                )
                saved += 1
            elif args.save_failure_videos and collector.episode.base_images:
                collector.episode.save_videos(
                    args.output_dir / "failure_videos", f"attempt_{attempt:06d}", cfg.control_freq
                )
    finally:
        collector.close()
    if saved < cfg.episodes:
        raise RuntimeError(f"only collected {saved}/{cfg.episodes} successful episodes")


if __name__ == "__main__":
    main()
