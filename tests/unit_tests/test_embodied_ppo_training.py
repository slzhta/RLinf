import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from rlinf.algorithms.losses import compute_decoupled_ppo_actor_critic_loss
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    index_flattened_batch,
    make_rollout_permutation,
)


def test_rollout_permutation_changes_across_steps_and_epochs():
    kwargs = {"rollout_size": 32, "seed": 7, "rank": 0, "update_epoch": 4}
    first = make_rollout_permutation(global_step=3, epoch=0, **kwargs)
    repeated = make_rollout_permutation(global_step=3, epoch=0, **kwargs)
    next_epoch = make_rollout_permutation(global_step=3, epoch=1, **kwargs)
    next_step = make_rollout_permutation(global_step=4, epoch=0, **kwargs)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_epoch)
    assert not torch.equal(first, next_step)
    assert torch.equal(first.sort().values, torch.arange(32))


def test_index_flattened_batch_uses_one_permutation_for_nested_tensors():
    batch = {
        "actions": torch.arange(6),
        "forward_inputs": {"states": torch.arange(12).reshape(6, 2)},
        "optional": None,
    }
    indices = torch.tensor([5, 1, 3, 0, 4, 2])

    shuffled = index_flattened_batch(batch, indices)

    assert torch.equal(shuffled["actions"], batch["actions"][indices])
    assert torch.equal(
        shuffled["forward_inputs"]["states"],
        batch["forward_inputs"]["states"][indices],
    )
    assert shuffled["optional"] is None


def test_value_loss_coefficient_scales_actor_critic_loss():
    shape = (8, 1)
    kwargs = {
        "logprobs": torch.zeros(shape, dtype=torch.float32),
        "old_logprobs": torch.zeros(shape, dtype=torch.float32),
        "advantages": torch.zeros(shape, dtype=torch.float32),
        "values": torch.zeros(shape, dtype=torch.float32),
        "returns": torch.ones(shape, dtype=torch.float32),
        "prev_values": torch.zeros(shape, dtype=torch.float32),
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "value_clip": 0.2,
        "huber_delta": 10.0,
        "loss_mask": torch.ones(shape, dtype=torch.bool),
    }

    full_loss, _ = compute_decoupled_ppo_actor_critic_loss(
        value_loss_coef=1.0, **kwargs
    )
    half_loss, _ = compute_decoupled_ppo_actor_critic_loss(
        value_loss_coef=0.5, **kwargs
    )

    assert torch.isclose(half_loss, 0.5 * full_loss)


class _IdentityGradScaler:
    def unscale_(self, optimizer):
        del optimizer

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


class _ClipStrategy:
    def clip_grad_norm_(self, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)


class _WarningLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)


class _TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(1, 1, bias=False)
        self.value_head = torch.nn.Linear(1, 1, bias=False)


class _TinyVisualActorCritic(_TinyActorCritic):
    def __init__(self):
        super().__init__()
        self.resnet_backbone = torch.nn.Linear(1, 1, bias=False)


def test_optimizer_uses_smaller_visual_backbone_learning_rate():
    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 3.0e-4,
                "backbone_lr": 3.0e-5,
                "value_lr": 2.0e-4,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
            }
        }
    )

    optimizer = manager.build_optimizer(_TinyVisualActorCritic())

    assert [group["group_type"] for group in optimizer.param_groups] == [
        "actor",
        "critic",
        "actor_backbone",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [
        3.0e-4,
        2.0e-4,
        3.0e-5,
    ]


def test_critic_warmup_keeps_optimizer_stable_and_freezes_actor_group():
    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 0.1,
                "value_lr": 0.1,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_eps": 1.0e-8,
                "weight_decay": 0.0,
            }
        }
    )
    manager._actor_optimizer_group_indices = []
    manager.optimizer_steps = 0
    manager.critic_warmup_steps = 1
    manager.model = _TinyActorCritic()
    manager.optimizer = manager.build_optimizer(manager.model)
    manager.grad_scaler = _IdentityGradScaler()
    manager._strategy = _ClipStrategy()

    optimizer_id = id(manager.optimizer)
    actor_before = manager.model.actor.weight.detach().clone()
    value_before = manager.model.value_head.weight.detach().clone()
    (
        manager.model.actor.weight.sum() + manager.model.value_head.weight.sum()
    ).backward()

    _, warmup_lrs = manager.optimizer_step()

    assert id(manager.optimizer) == optimizer_id
    assert warmup_lrs[0] == 0.0
    assert torch.equal(manager.model.actor.weight, actor_before)
    assert not torch.equal(manager.model.value_head.weight, value_before)

    manager.optimizer.zero_grad()
    actor_before = manager.model.actor.weight.detach().clone()
    (
        manager.model.actor.weight.sum() + manager.model.value_head.weight.sum()
    ).backward()
    manager.optimizer_step()

    assert id(manager.optimizer) == optimizer_id
    assert not torch.equal(manager.model.actor.weight, actor_before)


def test_legacy_critic_only_scheduler_is_migrated_without_actor_lr_jump():
    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 2.0e-5,
                "value_lr": 2.0e-4,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "total_training_steps": 1000,
            }
        }
    )
    actor = torch.nn.Parameter(torch.ones(()))
    critic = torch.nn.Parameter(torch.ones(()))
    manager.optimizer = torch.optim.AdamW(
        [
            {"params": [actor], "lr": manager._cfg.optim.lr},
            {"params": [critic], "lr": manager._cfg.optim.value_lr},
        ]
    )
    manager._actor_optimizer_group_indices = [0]
    manager.lr_scheduler = manager.build_lr_scheduler(
        manager.optimizer, manager._cfg.optim
    )
    for _ in range(10):
        manager.optimizer.step()
        manager.lr_scheduler.step()
    last_epoch = manager.lr_scheduler.last_epoch
    step_count = manager.lr_scheduler._step_count
    manager.lr_scheduler.base_lrs = [manager._cfg.optim.value_lr]
    manager.lr_scheduler._last_lr = [manager._cfg.optim.value_lr]
    manager._logger = _WarningLogger()

    manager._repair_legacy_lr_scheduler()

    assert manager.lr_scheduler.base_lrs == [2.0e-5, 2.0e-4]
    assert manager.lr_scheduler.last_epoch == last_epoch
    assert manager.lr_scheduler._step_count == step_count
    manager.optimizer.step()
    manager.lr_scheduler.step()
    assert [group["lr"] for group in manager.optimizer.param_groups] == [
        2.0e-5,
        2.0e-4,
    ]
    assert len(manager._logger.messages) == 1


def test_ppo_resume_rejects_bc_checkpoint_before_optimizer_load(tmp_path):
    torch.save(
        {"format_version": 2, "training_stage": "bc", "optimizer_steps": 10},
        tmp_path / "trainer_state.pt",
    )
    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create({"checkpoint_stage": "ppo"})

    with pytest.raises(ValueError, match="not a 'ppo' checkpoint"):
        manager.load_checkpoint(str(tmp_path))


def test_ppo_resume_rejects_action_distribution_change(tmp_path):
    torch.save(
        {
            "format_version": 2,
            "training_stage": "ppo",
            "action_distribution_version": 1,
            "binary_action_indices": (6,),
            "binary_action_temperature": 1.0,
        },
        tmp_path / "trainer_state.pt",
    )
    manager = object.__new__(FSDPModelManager)
    manager._cfg = OmegaConf.create(
        {
            "checkpoint_stage": "ppo",
            "model": {
                "binary_action_indices": [6],
                "binary_action_temperature": 0.2,
            },
        }
    )

    with pytest.raises(ValueError, match="incompatible action distribution"):
        manager.load_checkpoint(str(tmp_path))


def test_ppo_initialization_accepts_only_matching_bc_semantics(tmp_path):
    actor_dir = tmp_path / "actor"
    checkpoint_path = actor_dir / "model_state_dict" / "full_weights.pt"
    checkpoint_path.parent.mkdir(parents=True)
    marker = {
        "format_version": 2,
        "training_stage": "bc",
        "model_type": "cnn_policy",
        "binary_action_indices": (6,),
        "binary_action_temperature": 1.0,
        "binary_loss": "bce_with_logits",
        "continuous_action_distribution": "tanh_normal_v1",
        "state_mean": (1.0, 2.0),
        "state_std": (3.0, 4.0),
    }
    torch.save(marker, actor_dir / "bc_policy_state.pt")
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "actor": {
                "initial_checkpoint_stage": "bc",
                "model": {
                    "model_type": "cnn_policy",
                    "binary_action_indices": [6],
                    "binary_action_temperature": 1.0,
                    "state_mean": [1.0, 2.0],
                    "state_std": [3.0, 4.0],
                },
            }
        }
    )

    actor._validate_initial_model_checkpoint(str(checkpoint_path))
    actor.cfg.actor.model.state_std = [3.0, 5.0]
    with pytest.raises(ValueError, match="incompatible"):
        actor._validate_initial_model_checkpoint(str(checkpoint_path))


def test_ppo_initialization_can_keep_configured_logstd(tmp_path, monkeypatch):
    class TinyPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_mean = nn.Parameter(torch.tensor([0.0]))
            self.actor_logstd = nn.Parameter(torch.tensor([-1.8]))

    checkpoint_path = tmp_path / "full_weights.pt"
    torch.save(
        {
            "actor_mean": torch.tensor([0.75]),
            "actor_logstd": torch.tensor([-2.0]),
        },
        checkpoint_path,
    )
    policy = TinyPolicy()
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.get_model", lambda _: policy
    )
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"ckpt_path": str(checkpoint_path)},
            "actor": {
                "initial_checkpoint_exclude_keys": ["actor_logstd"],
                "model": {},
            },
        }
    )

    loaded_policy = actor.model_provider_func()

    assert loaded_policy.actor_mean.item() == pytest.approx(0.75)
    assert loaded_policy.actor_logstd.item() == pytest.approx(-1.8)


class _MetricLogger:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, data, step):
        self.logged.append((data, step))

    def finish(self):
        self.finished = True


class _Joinable:
    def join(self, timeout=None):
        del timeout


def test_embodied_runner_records_pretraining_evaluation():
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"eval_before_training": True}})
    runner.global_step = 7
    runner.max_steps = 0
    runner.metric_logger = _MetricLogger()
    runner.log_queue = _Joinable()
    runner.log_thread = _Joinable()
    runner.stop_logging = False
    calls = []
    runner.update_rollout_weights = lambda: calls.append("sync")
    runner.evaluate = lambda: {"success_once": 0.625, "num_trajectories": 128}

    runner.run()

    assert calls == ["sync"]
    assert runner.metric_logger.logged == [
        (
            {
                "eval/success_once": 0.625,
                "eval/num_trajectories": 128,
            },
            7,
        )
    ]
    assert runner.metric_logger.finished
