import pytest
import torch

from rlinf.envs.maniskill.tasks.digital_twin.scripted_pick_and_place_expert import (
    PickAndPlacePhase,
    ScriptedPickAndPlaceExpert,
    validate_persistent_gripper_commands,
)


def _actions(gripper_commands: list[float]) -> torch.Tensor:
    actions = torch.zeros(len(gripper_commands), 1, 7)
    actions[:, 0, 6] = torch.tensor(gripper_commands)
    return actions


def test_persistent_gripper_command_validation():
    stats = validate_persistent_gripper_commands(
        _actions([1.0] * 3 + [-1.0] * 5 + [1.0] * 2)
    )

    assert stats == {
        "episode_steps": 10,
        "open_steps": 5,
        "close_steps": 5,
        "close_run_steps": 5,
    }


def test_gripper_pulse_is_rejected():
    with pytest.raises(ValueError, match="pulse labels are invalid"):
        validate_persistent_gripper_commands(_actions([1.0] * 3 + [-1.0] + [1.0] * 3))


def test_expert_action_is_deterministic_for_same_state():
    experts = [ScriptedPickAndPlaceExpert(num_envs=1, device="cpu") for _ in range(2)]
    cube = torch.tensor([[0.50, 0.08, 0.03]])
    goal = torch.tensor([[0.50, -0.08, 0.03]])
    tcp = torch.tensor([[0.45, 0.00, 0.12]])
    false = torch.tensor([False])

    actions = []
    for expert in experts:
        expert.reset(cube, goal)
        expert.phase.fill_(int(PickAndPlacePhase.APPROACH))
        actions.append(expert.act(tcp, cube, goal, false, false, false, false))

    torch.testing.assert_close(actions[0], actions[1])
    assert actions[0][0, 6] == 1.0


def test_expert_grasp_and_release_transitions_are_feedback_reactive():
    expert = ScriptedPickAndPlaceExpert(num_envs=1, device="cpu")
    cube = torch.tensor([[0.50, 0.08, 0.03]])
    goal = torch.tensor([[0.50, -0.08, 0.03]])
    false = torch.tensor([False])
    true = torch.tensor([True])
    expert.reset(cube, goal)
    targets = expert._targets(cube, goal)

    expert.phase.fill_(int(PickAndPlacePhase.GRASP))
    grasp_action = expert.act(
        targets[PickAndPlacePhase.GRASP],
        cube,
        goal,
        true,
        true,
        false,
        false,
    )
    assert expert.phase.item() == int(PickAndPlacePhase.LIFT)
    assert grasp_action[0, 6] == -1.0

    expert.phase.fill_(int(PickAndPlacePhase.RELEASE))
    release_action = expert.act(
        targets[PickAndPlacePhase.RELEASE],
        cube,
        goal,
        false,
        true,
        true,
        false,
    )
    assert expert.phase.item() == int(PickAndPlacePhase.RETREAT)
    assert release_action[0, 6] == 1.0
