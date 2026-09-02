from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class PickAndPlacePhase(IntEnum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    TRANSIT = 4
    ABOVE_GOAL = 5
    LOWER = 6
    RELEASE = 7
    RETREAT = 8
    DONE = 9


@dataclass
class ScriptedPickAndPlaceExpertConfig:
    position_action_scale: float = 0.01
    control_gain: float = 0.35
    max_position_action: float = 0.75
    position_tolerance: float = 0.012
    grasp_position_tolerance: float = 0.007
    hover_height: float = 0.0925
    transit_height: float = 0.1225
    grasp_tcp_z_offset: float = 0.011
    place_tcp_z_offset: float = 0.013


def validate_persistent_gripper_commands(actions: torch.Tensor) -> dict[str, int]:
    """Validate persistent open-close-open labels for one successful episode."""
    if actions.ndim != 3 or actions.shape[1] != 1 or actions.shape[2] != 7:
        raise ValueError(f"Expected trajectory actions [T, 1, 7], got {actions.shape}.")

    gripper = actions[:, 0, 6]
    is_open = gripper == 1.0
    is_closed = gripper == -1.0
    if not torch.all(is_open | is_closed):
        raise ValueError("Gripper labels must contain only persistent -1/+1 commands.")

    transitions = torch.nonzero(gripper[1:] != gripper[:-1]).flatten() + 1
    if len(transitions) != 2 or not is_open[0] or not is_open[-1]:
        raise ValueError(
            "A successful demonstration must follow one open-close-open command cycle."
        )

    close_start = int(transitions[0].item())
    close_end = int(transitions[1].item())
    close_steps = close_end - close_start
    if close_steps < 4:
        raise ValueError(
            f"Close command lasted only {close_steps} steps; pulse labels are invalid."
        )
    return {
        "episode_steps": int(len(gripper)),
        "open_steps": int(is_open.sum().item()),
        "close_steps": int(is_closed.sum().item()),
        "close_run_steps": close_steps,
    }


class ScriptedPickAndPlaceExpert:
    """Deterministic privileged-state feedback expert for PnP demonstrations."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: ScriptedPickAndPlaceExpertConfig | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = config or ScriptedPickAndPlaceExpertConfig()
        self.phase = torch.full(
            (self.num_envs,),
            int(PickAndPlacePhase.APPROACH),
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, cube_positions: torch.Tensor, goal_positions: torch.Tensor) -> None:
        if cube_positions.shape != (self.num_envs, 3):
            raise ValueError(
                f"Expected cube positions {(self.num_envs, 3)}, got {cube_positions.shape}."
            )
        if goal_positions.shape != (self.num_envs, 3):
            raise ValueError(
                f"Expected goal positions {(self.num_envs, 3)}, got {goal_positions.shape}."
            )

        self.source_positions = cube_positions.to(self.device).clone()
        self.goal_positions = goal_positions.to(self.device).clone()

        self.phase.fill_(int(PickAndPlacePhase.APPROACH))

    def _set_phase(self, mask: torch.Tensor, phase: PickAndPlacePhase) -> None:
        if not torch.any(mask):
            return
        self.phase[mask] = int(phase)

    def _targets(
        self, cube_positions: torch.Tensor, goal_positions: torch.Tensor
    ) -> dict[PickAndPlacePhase, torch.Tensor]:
        cfg = self.config
        grasp_xy = cube_positions[:, :2]
        place_xy = goal_positions[:, :2]

        approach = torch.cat(
            [grasp_xy, (cube_positions[:, 2] + cfg.hover_height).unsqueeze(1)],
            dim=1,
        )
        descend = torch.cat(
            [grasp_xy, (cube_positions[:, 2] + cfg.grasp_tcp_z_offset).unsqueeze(1)],
            dim=1,
        )
        transit_z = goal_positions[:, 2] + cfg.transit_height
        midpoint_xy = 0.5 * (self.source_positions[:, :2] + goal_positions[:, :2])
        transit = torch.cat([midpoint_xy, transit_z.unsqueeze(1)], dim=1)
        above_goal = torch.cat([place_xy, transit_z.unsqueeze(1)], dim=1)
        lower = torch.cat(
            [place_xy, (goal_positions[:, 2] + cfg.place_tcp_z_offset).unsqueeze(1)],
            dim=1,
        )
        retreat = torch.cat(
            [place_xy, (goal_positions[:, 2] + cfg.hover_height).unsqueeze(1)],
            dim=1,
        )
        lift = torch.cat([grasp_xy, transit_z.unsqueeze(1)], dim=1)
        return {
            PickAndPlacePhase.APPROACH: approach,
            PickAndPlacePhase.DESCEND: descend,
            PickAndPlacePhase.GRASP: descend,
            PickAndPlacePhase.LIFT: lift,
            PickAndPlacePhase.TRANSIT: transit,
            PickAndPlacePhase.ABOVE_GOAL: above_goal,
            PickAndPlacePhase.LOWER: lower,
            PickAndPlacePhase.RELEASE: lower,
            PickAndPlacePhase.RETREAT: retreat,
            PickAndPlacePhase.DONE: retreat,
        }

    def _advance_phases(
        self,
        tcp_positions: torch.Tensor,
        targets: dict[PickAndPlacePhase, torch.Tensor],
        is_grasped: torch.Tensor,
        is_lifted: torch.Tensor,
        is_placed: torch.Tensor,
        success: torch.Tensor,
    ) -> None:
        cfg = self.config

        def near(phase: PickAndPlacePhase, tolerance: float) -> torch.Tensor:
            distance = torch.linalg.norm(targets[phase] - tcp_positions, dim=1)
            return (self.phase == int(phase)) & (distance <= tolerance)

        self._set_phase(
            near(PickAndPlacePhase.APPROACH, cfg.position_tolerance),
            PickAndPlacePhase.DESCEND,
        )
        self._set_phase(
            near(PickAndPlacePhase.DESCEND, cfg.grasp_position_tolerance),
            PickAndPlacePhase.GRASP,
        )

        # Grasp/release transitions use measured contact state instead of hidden
        # dwell counters, so the demonstrations remain feedback-reactive.
        grasp_complete = (self.phase == int(PickAndPlacePhase.GRASP)) & is_grasped
        self._set_phase(grasp_complete, PickAndPlacePhase.LIFT)

        lift_complete = (
            (self.phase == int(PickAndPlacePhase.LIFT))
            & is_grasped
            & is_lifted
            & near(PickAndPlacePhase.LIFT, cfg.position_tolerance)
        )
        self._set_phase(lift_complete, PickAndPlacePhase.TRANSIT)
        self._set_phase(
            near(PickAndPlacePhase.TRANSIT, cfg.position_tolerance) & is_grasped,
            PickAndPlacePhase.ABOVE_GOAL,
        )
        self._set_phase(
            near(PickAndPlacePhase.ABOVE_GOAL, cfg.position_tolerance) & is_grasped,
            PickAndPlacePhase.LOWER,
        )
        lower_complete = (
            (self.phase == int(PickAndPlacePhase.LOWER))
            & is_grasped
            & (is_placed | near(PickAndPlacePhase.LOWER, cfg.grasp_position_tolerance))
        )
        self._set_phase(lower_complete, PickAndPlacePhase.RELEASE)
        release_complete = (self.phase == int(PickAndPlacePhase.RELEASE)) & (
            ~is_grasped
        )
        self._set_phase(release_complete, PickAndPlacePhase.RETREAT)
        self._set_phase(success, PickAndPlacePhase.DONE)

    def act(
        self,
        tcp_positions: torch.Tensor,
        cube_positions: torch.Tensor,
        goal_positions: torch.Tensor,
        is_grasped: torch.Tensor,
        is_lifted: torch.Tensor,
        is_placed: torch.Tensor,
        success: torch.Tensor,
    ) -> torch.Tensor:
        tcp_positions = tcp_positions.to(self.device)
        cube_positions = cube_positions.to(self.device)
        goal_positions = goal_positions.to(self.device)
        is_grasped = is_grasped.to(self.device, dtype=torch.bool)
        is_lifted = is_lifted.to(self.device, dtype=torch.bool)
        is_placed = is_placed.to(self.device, dtype=torch.bool)
        success = success.to(self.device, dtype=torch.bool)

        targets = self._targets(cube_positions, goal_positions)
        self._advance_phases(
            tcp_positions,
            targets,
            is_grasped,
            is_lifted,
            is_placed,
            success,
        )
        targets = self._targets(cube_positions, goal_positions)

        target_positions = torch.zeros_like(tcp_positions)
        for phase in PickAndPlacePhase:
            mask = self.phase == int(phase)
            target_positions[mask] = targets[phase][mask]

        cfg = self.config
        delta = target_positions - tcp_positions
        position_action = delta / cfg.position_action_scale * cfg.control_gain
        position_action = torch.clamp(
            position_action,
            min=-cfg.max_position_action,
            max=cfg.max_position_action,
        )
        position_action = torch.clamp(position_action, -1.0, 1.0)

        actions = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=self.device
        )
        actions[:, :3] = position_action
        # The gripper label is a persistent desired state, not a one-step trigger.
        keep_closed = (self.phase >= int(PickAndPlacePhase.GRASP)) & (
            self.phase < int(PickAndPlacePhase.RELEASE)
        )
        actions[:, 6] = torch.where(
            keep_closed,
            torch.full_like(actions[:, 6], -1.0),
            torch.full_like(actions[:, 6], 1.0),
        )
        done_mask = self.phase == int(PickAndPlacePhase.DONE)
        actions[done_mask, :6] = 0.0
        return actions
