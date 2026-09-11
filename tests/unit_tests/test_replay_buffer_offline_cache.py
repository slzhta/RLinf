import pytest
import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _trajectory(value: float) -> Trajectory:
    actions = torch.full((2, 1, 2), value)
    rewards = torch.zeros((2, 1, 1))
    return Trajectory(
        max_episode_length=2,
        model_weights_id="expert",
        actions=actions,
        rewards=rewards,
        terminations=torch.zeros_like(rewards, dtype=torch.bool),
        truncations=torch.zeros_like(rewards, dtype=torch.bool),
        dones=torch.zeros_like(rewards, dtype=torch.bool),
        forward_inputs={
            "states": torch.full((2, 1, 3), value),
            "action": actions.clone(),
            "sample_weight": torch.tensor([[[1.0]], [[4.0]]]),
        },
    )


def test_read_only_checkpoint_honors_cache_size(tmp_path):
    source = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(tmp_path),
        enable_cache=False,
        cache_size=0,
        sample_window_size=3,
    )
    source.add_trajectories([_trajectory(1.0), _trajectory(2.0), _trajectory(3.0)])
    source.close()

    loaded = TrajectoryReplayBuffer(
        auto_save=False,
        enable_cache=True,
        cache_size=1,
        sample_window_size=3,
        cache_all_when_not_saving=False,
    )
    loaded.load_checkpoint(str(tmp_path))

    assert loaded._flat_trajectory_cache.max_size == 1
    assert loaded.get_stats()["cache_size"] == 1
    assert loaded.get_stats()["cache_bytes"] > 0
    assert loaded.sample(2)["forward_inputs"]["states"].shape == (2, 3)
    loaded.close()


def test_disk_cache_clamps_capacity_and_reserves_final_slot_length(tmp_path):
    source = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(tmp_path),
        enable_cache=False,
        sample_window_size=0,
    )
    source.add_trajectories([_trajectory(1.0), _trajectory(2.0), _trajectory(3.0)])
    source.close()

    loaded = TrajectoryReplayBuffer(
        auto_save=False,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=0,
        cache_all_when_not_saving=False,
    )
    loaded.load_checkpoint(str(tmp_path), restore_seed=False)

    cache = loaded._flat_trajectory_cache
    assert cache.max_size == 3
    assert cache.get_slot_length() == 2
    assert loaded.get_stats()["cache_size"] == 3
    loaded.sample(1)
    assert loaded._window_cache_ids == [0, 1, 2]
    loaded.close()


def test_weighted_sampling_is_unique_and_restorable():
    replay = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        sample_window_size=2,
    )
    replay.add_trajectories([_trajectory(1.0), _trajectory(2.0)])

    sampling_state = replay.sampling_state_dict()
    first_batch = replay.sample(4, sampling_weight_key="forward_inputs.sample_weight")
    replay.load_sampling_state_dict(sampling_state)
    repeated_batch = replay.sample(
        4, sampling_weight_key="forward_inputs.sample_weight"
    )

    torch.testing.assert_close(first_batch["actions"], repeated_batch["actions"])
    # Each source transition is selected once because weighted batches are
    # sampled without replacement.
    assert sorted(
        first_batch["forward_inputs"]["sample_weight"].flatten().tolist()
    ) == [1.0, 1.0, 4.0, 4.0]
    with pytest.raises(ValueError, match="unique transitions"):
        replay.sample(5, sampling_weight_key="forward_inputs.sample_weight")
    replay.close()


def test_weighted_sampling_changes_selection_probability():
    replay = TrajectoryReplayBuffer(
        seed=11,
        enable_cache=False,
        sample_window_size=2,
    )
    replay.add_trajectories([_trajectory(1.0), _trajectory(2.0)])

    high_weight = 0
    trials = 1000
    for _ in range(trials):
        batch = replay.sample(1, sampling_weight_key="forward_inputs.sample_weight")
        high_weight += int(batch["forward_inputs"]["sample_weight"].item() == 4.0)

    assert 0.75 < high_weight / trials < 0.85
    replay.close()


def test_unweighted_epoch_sampling_avoids_repeats_and_restores_cursor():
    replay = TrajectoryReplayBuffer(
        seed=19,
        enable_cache=True,
        sample_window_size=2,
    )
    replay.add_trajectories([_trajectory(1.0), _trajectory(2.0)])

    first = replay.sample(2, without_replacement=True)
    sampling_state = replay.sampling_state_dict()
    second = replay.sample(2, without_replacement=True)
    replay.load_sampling_state_dict(sampling_state)
    repeated_second = replay.sample(2, without_replacement=True)

    sample_ids = list(
        zip(
            torch.cat([first["actions"], second["actions"]])[:, 0].tolist(),
            torch.cat(
                [
                    first["forward_inputs"]["sample_weight"],
                    second["forward_inputs"]["sample_weight"],
                ]
            )[:, 0].tolist(),
            strict=True,
        )
    )
    assert len(set(sample_ids)) == 4
    torch.testing.assert_close(second["actions"], repeated_second["actions"])
    torch.testing.assert_close(
        second["forward_inputs"]["sample_weight"],
        repeated_second["forward_inputs"]["sample_weight"],
    )
    replay.close()
