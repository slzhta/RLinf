import json

import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.collect_maniskill_pnp_demos import (
    DemoDiversityTracker,
    build_action_sampling_metadata,
    validate_demonstration_trajectory,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


def test_binary_action_transitions_receive_sampling_weight():
    actions = torch.zeros(6, 1, 7)
    actions[:, :, 6] = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0, 1.0]).view(-1, 1)

    weights, transitions = build_action_sampling_metadata(actions, 20.0)

    assert transitions[:, 0, 0].tolist() == [False, False, True, False, True, False]
    assert weights[:, 0, 0].tolist() == [1.0, 1.0, 20.0, 1.0, 20.0, 1.0]


def test_diversity_tracker_rejects_quantized_near_duplicates():
    tracker = DemoDiversityTracker(
        initial_state_resolution=0.01,
        action_resolution=0.1,
        reject_near_duplicates=True,
    )
    initial_state = torch.tensor([0.50, 0.08, 0.03, 0.45, 0.00, 0.12])
    actions = torch.zeros(4, 1, 7)

    assert tracker.consider(initial_state, actions) == (True, None)
    accepted, reason = tracker.consider(initial_state + 0.001, actions + 0.001)

    assert not accepted
    assert reason == "initial_state"
    summary = tracker.summary()
    assert summary["candidate_episodes"] == 2
    assert summary["accepted_episodes"] == 1
    assert summary["accepted_candidate_ratio"] == 0.5
    assert summary["unique_candidate_ratio"] == 0.5


def test_diversity_tracker_distinguishes_effective_variation():
    tracker = DemoDiversityTracker(0.01, 0.1, True)
    initial_state = torch.zeros(6)
    actions = torch.zeros(4, 1, 7)
    varied_state = initial_state.clone()
    varied_state[0] = 0.02
    varied_actions = actions.clone()
    varied_actions[1, 0, 0] = 0.2

    assert tracker.consider(initial_state, actions)[0]
    assert tracker.consider(varied_state, varied_actions)[0]
    assert tracker.summary()["unique_trajectory_fingerprints"] == 2


def test_diversity_tracker_detects_near_states_across_quantization_boundary():
    tracker = DemoDiversityTracker(0.01, 0.1, True)
    actions = torch.zeros(4, 1, 7)

    assert tracker.consider(torch.tensor([0.0049]), actions)[0]
    accepted, reason = tracker.consider(torch.tensor([0.0051]), actions + 0.2)

    assert not accepted
    assert reason == "initial_state"


def test_diversity_summary_reports_state_action_coverage():
    tracker = DemoDiversityTracker(0.01, 0.1, True)
    actions = torch.zeros(2, 1, 7)
    actions[1, 0, 0] = 0.2
    task_states = torch.zeros(2, 1, 14)
    task_states[1, 0, 0] = 0.02

    assert tracker.consider(torch.zeros(14), actions, task_states)[0]
    summary = tracker.summary()

    assert summary["unique_quantized_state_action_transitions"] == 2
    assert summary["quantized_state_action_ratio"] == 1.0


def test_demonstration_validation_rejects_invalid_rgb():
    length = 2
    done = torch.tensor([False, True]).view(length, 1, 1)
    trajectory = Trajectory(
        max_episode_length=length,
        model_weights_id="expert",
        actions=torch.zeros(length, 1, 7),
        rewards=torch.zeros(length, 1, 1),
        terminations=done.clone(),
        truncations=torch.zeros_like(done),
        dones=done,
        forward_inputs={
            "main_images": torch.zeros(length, 1, 4, 4, 3),
            "states": torch.zeros(length, 1, 14),
            "task_states": torch.zeros(length, 1, 14),
            "action": torch.zeros(length, 1, 7),
            "sample_weight": torch.ones(length, 1, 1),
            "action_transition": torch.zeros(length, 1, 1, dtype=torch.bool),
        },
    )

    with pytest.raises(ValueError, match="uint8"):
        validate_demonstration_trajectory(trajectory)

    trajectory.forward_inputs["main_images"] = torch.zeros(
        length, 1, 4, 4, 3, dtype=torch.uint8
    )
    trajectory.actions[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="executed actions"):
        validate_demonstration_trajectory(trajectory)


def test_demonstration_validation_recomputes_transition_flags():
    length = 3
    actions = torch.zeros(length, 1, 7)
    actions[:, :, 6] = torch.tensor([1.0, -1.0, -1.0]).view(-1, 1)
    weights, transitions = build_action_sampling_metadata(actions, 5.0)
    done = torch.tensor([False, False, True]).view(length, 1, 1)
    trajectory = Trajectory(
        max_episode_length=length,
        model_weights_id="expert",
        actions=actions.clone(),
        rewards=torch.zeros(length, 1, 1),
        terminations=done.clone(),
        truncations=torch.zeros_like(done),
        dones=done,
        forward_inputs={
            "main_images": torch.zeros(length, 1, 4, 4, 3, dtype=torch.uint8),
            "states": torch.zeros(length, 1, 14),
            "task_states": torch.zeros(length, 1, 14),
            "action": actions,
            "sample_weight": weights,
            "action_transition": transitions,
        },
    )

    validate_demonstration_trajectory(trajectory)
    trajectory.forward_inputs["action_transition"][1] = False
    with pytest.raises(ValueError, match="transition metadata"):
        validate_demonstration_trajectory(trajectory)


def test_bc_rejects_dataset_manifest_replay_mismatch(tmp_path):
    replay_dir = tmp_path / "demos"
    replay_dir.mkdir()
    manifest_path = tmp_path / "collection_summary.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "collected_episodes": 1000,
                "total_transitions": 75000,
                "bc_sampling": {"weight_key": "forward_inputs.sample_weight"},
            }
        )
    )
    (replay_dir / "metadata.json").write_text(
        json.dumps({"size": 999, "total_samples": 75000})
    )
    worker = object.__new__(EmbodiedDAGGERFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {
                    "manifest_path": str(manifest_path),
                    "required_data_format_version": 2,
                    "required_num_trajectories": 1000,
                }
            }
        }
    )
    worker._replay_sampling_weight_key = "forward_inputs.sample_weight"

    with pytest.raises(ValueError, match="replay size"):
        worker._validate_offline_dataset_manifest(str(replay_dir))


def test_bc_resume_rejects_checkpoint_without_format_marker(tmp_path):
    worker = object.__new__(EmbodiedDAGGERFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "binary_action_indices": [6],
                    "binary_action_temperature": 1.0,
                }
            }
        }
    )
    worker._offline_data_format_version = 2

    with pytest.raises(ValueError, match="no format marker"):
        worker._validate_bc_checkpoint_marker(str(tmp_path))
