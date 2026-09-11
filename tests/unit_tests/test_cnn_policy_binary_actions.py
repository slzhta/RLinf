from types import SimpleNamespace

import torch

from rlinf.models.embodiment.cnn_policy.cnn_policy import CNNPolicy


def _hybrid_policy() -> CNNPolicy:
    policy = object.__new__(CNNPolicy)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(
        action_dim=7,
        num_action_chunks=1,
        action_std_scale=[1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0],
        std_range=None,
        logstd_range=(-3.0, -1.5),
        binary_action_temperature=1.0,
        bc_smooth_l1_beta=0.1,
        bc_action_loss_weights=[],
    )
    policy._binary_action_indices = (6,)
    policy._continuous_action_indices = (0, 1, 2, 3, 4, 5)
    policy.action_scale = None
    return policy


def test_binary_action_metrics_include_transition_recall():
    predicted_open = torch.tensor([[True], [False], [True], [False]])
    target_open = torch.tensor([[True], [False], [False], [True]])
    transitions = torch.tensor([[False], [True], [True], [True]])

    counts = CNNPolicy._binary_sft_counts(predicted_open, target_open, transitions)

    assert counts["binary_accuracy_correct"] == 2
    assert counts["binary_accuracy_count"] == 4
    assert counts["binary_close_recall_correct"] == 1
    assert counts["binary_close_recall_count"] == 2
    assert counts["binary_transition_close_recall_correct"] == 1
    assert counts["binary_transition_close_recall_count"] == 2
    assert counts["binary_transition_open_recall_correct"] == 0
    assert counts["binary_transition_open_recall_count"] == 1


def test_hybrid_action_logprobs_match_rollout_actions():
    policy = _hybrid_policy()
    action_mean = torch.tensor(
        [[0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.8]], requires_grad=True
    )
    action_logstd = torch.full_like(action_mean, -2.0)
    action_std = policy._action_std(action_logstd)
    action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    logprobs, entropy = policy._hybrid_action_statistics(
        action_mean, action_std, action
    )
    (logprobs.sum() + entropy.sum()).backward()

    assert logprobs.shape == action.shape
    assert entropy.shape == action.shape
    assert torch.isclose(action_std[0, 3], torch.exp(torch.tensor(-2.0)) * 0.1)
    assert action_mean.grad is not None
    assert torch.all(torch.isfinite(action_mean.grad))
    assert action_mean.grad[0, 6] != 0


def test_hybrid_eval_emits_exact_binary_commands():
    policy = _hybrid_policy()
    action_mean = torch.tensor(
        [[0.1, -0.1, 0.2, 0.0, 0.0, 0.0, -0.25], [0.0] * 6 + [0.25]]
    )
    action_logstd = torch.full_like(action_mean, -2.0)
    policy._actor_forward_from_processed_tensors = lambda **_: (
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        action_mean,
        action_logstd,
    )

    action, chunk_actions, rollout_logprobs, _, _ = policy._generate_actions(
        states=None,
        main_images=torch.zeros(2, 1),
        extra_view_images=None,
        calculate_values=False,
        mode="eval",
    )
    recomputed_logprobs, _ = policy._hybrid_action_statistics(
        action_mean, policy._action_std(action_logstd), action
    )

    assert action[:, 6].tolist() == [-1.0, 1.0]
    assert torch.equal(chunk_actions[:, 0], action)
    torch.testing.assert_close(rollout_logprobs, recomputed_logprobs)


def test_hybrid_train_actions_are_bounded_and_recomputable():
    policy = _hybrid_policy()
    action_mean = torch.tensor([[4.0, -4.0, 2.0, -2.0, 1.0, -1.0, 0.0]])
    action_logstd = torch.full_like(action_mean, -2.0)
    policy._actor_forward_from_processed_tensors = lambda **_: (
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        action_mean,
        action_logstd,
    )

    action, _, rollout_logprobs, _, _ = policy._generate_actions(
        states=None,
        main_images=torch.zeros(1, 1),
        extra_view_images=None,
        calculate_values=False,
        mode="train",
    )
    recomputed_logprobs, _ = policy._hybrid_action_statistics(
        action_mean, policy._action_std(action_logstd), action
    )

    assert torch.all(action[:, :6].abs() < 1.0)
    assert action[:, 6].item() in (-1.0, 1.0)
    torch.testing.assert_close(rollout_logprobs, recomputed_logprobs)


def test_state_preprocessing_uses_configured_normalization():
    policy = object.__new__(CNNPolicy)
    torch.nn.Module.__init__(policy)
    policy.cfg = SimpleNamespace(use_state=True)
    policy.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(())))
    policy.register_buffer(
        "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
    )
    policy.register_buffer(
        "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
    )
    policy.register_buffer("state_mean", torch.tensor([1.0, 1.0]))
    policy.register_buffer("state_std", torch.tensor([2.0, 4.0]))

    processed = policy.preprocess_env_obs(
        {
            "main_images": torch.zeros(1, 1, 1, 3, dtype=torch.uint8),
            "states": torch.tensor([[3.0, 5.0]]),
        }
    )

    torch.testing.assert_close(processed["states"], torch.ones(1, 2))


def test_low_binary_temperature_sharpens_persistent_gripper_command():
    policy = _hybrid_policy()
    policy.cfg.binary_action_temperature = 0.2
    action_mean = torch.zeros(1, 7)
    action_mean[0, 6] = 1.0
    action = torch.zeros(1, 7)
    action[0, 6] = 1.0

    logprobs, _ = policy._hybrid_action_statistics(
        action_mean,
        policy._action_std(torch.full_like(action_mean, -2.0)),
        action,
    )

    assert torch.exp(logprobs[0, 6]) > 0.99


def test_uniform_bc_loss_ignores_legacy_sampling_weights():
    policy = _hybrid_policy()
    predicted_actions = torch.zeros(2, 7, requires_grad=True)
    target_actions = torch.ones(2, 7)
    target_actions[1, 6] = -1.0
    policy.preprocess_env_obs = lambda obs: obs
    policy._actor_forward_from_processed_tensors = lambda **_: (
        None,
        None,
        predicted_actions,
        None,
    )

    loss, _ = policy.sft_forward(
        {
            "main_images": torch.zeros(2, 1),
            "action": target_actions,
            "sample_weight": torch.tensor([[1.0], [5.0]]),
        }
    )

    torch.testing.assert_close(loss[0, 0], loss[1, 0])
    torch.testing.assert_close(loss[0, 6], loss[1, 6])
