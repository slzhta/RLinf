"""Regression coverage for upstream command intervals with local reset fixes."""
from types import SimpleNamespace

from rlinf.envs.realworld.franka.franka_env import FrankaEnv


def make_env(monkeypatch, confirm_steps=1):
    env = object.__new__(FrankaEnv)
    env.config = SimpleNamespace(binary_gripper_threshold=0.5,
        use_zero_one_gripper_action=False, gripper_min_command_interval=1.0,
        gripper_open_confirm_steps=confirm_steps)
    env._franka_state = SimpleNamespace(gripper_open=False)
    env._logger = SimpleNamespace(info=lambda *args: None)
    env._last_gripper_command_time = float("-inf")
    env._gripper_open_command_count = 0
    calls = []
    def send(name):
        calls.append(name)
        return SimpleNamespace(wait=lambda: None)
    env._controller = SimpleNamespace(open_gripper=lambda: send("open"),
        close_gripper=lambda: send("close"))
    monkeypatch.setattr("rlinf.envs.realworld.franka.franka_env.time.sleep", lambda _: None)
    monkeypatch.setattr("rlinf.envs.realworld.franka.franka_env.time.monotonic", lambda: 10.0)
    return env, calls


def test_forced_reset_bypasses_stale_flag_cooldown_and_confirmation(monkeypatch):
    env, calls = make_env(monkeypatch, confirm_steps=3)
    env._franka_state.gripper_open = True
    env._last_gripper_command_time = 9.9
    assert env._gripper_action(1.0, force_command=True)
    assert calls == ["open"]
    assert env._last_gripper_command_time == 10.0


def test_confirmation_and_interval_both_apply_to_policy_actions(monkeypatch):
    env, calls = make_env(monkeypatch, confirm_steps=2)
    assert not env._gripper_action(1.0)
    assert calls == []
    assert env._gripper_action(1.0)
    env._franka_state.gripper_open = True
    assert not env._gripper_action(-1.0)
    assert calls == ["open"]


def test_gripper_constructor_retains_new_parameter_interface_and_local_defaults():
    import inspect
    from rlinf.envs.realworld.common.gripper.franka_gripper import FrankaGripper
    params = inspect.signature(FrankaGripper).parameters
    assert params["open_width"].default == 0.08
    assert params["epsilon_inner"].default == 0.005
    assert params["epsilon_outer"].default == 0.060
    assert "close_force" in params
