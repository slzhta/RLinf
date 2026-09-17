"""Hardware-free validation and export for six-axis, dual-view pouring demos."""

import numpy as np

from rlinf.utils.realworld_collection import numpy_copy


def validate_collection_config(cfg) -> None:
    """Reject configuration changes that would break the BC observation contract."""
    env = cfg.env.eval
    if env.init_params.id != "FrankaCoTrainingPourWaterEnv-v1":
        raise ValueError("Use the native Pour Water environment.")
    if env.auto_reset or env.ignore_terminations or env.no_gripper:
        raise ValueError(
            "Disable auto reset, ignored terminations and GripperCloseEnv."
        )
    if not env.use_spacemouse or env.spacemouse_gripper_enabled or env.use_gello:
        raise ValueError("Use six-axis SpaceMouse input without gripper commands.")
    if env.use_relative_frame or env.main_image_key != "wrist_2":
        raise ValueError("Use robot-base actions and third-view wrist_2 as main image.")
    if not env.include_states_in_obs or list(env.state_keys) != [
        "arm_joint_position",
        "tcp_pose",
        "gripper_open_state",
    ]:
        raise ValueError("Pour requires joint7 + TCP xyz/Euler6 + fixed gripper state.")
    cameras = list(env.override_cfg.camera_serials)
    if (
        len(cameras) != 2
        or len(set(cameras)) != 2
        or any(not str(s) or str(s).startswith("REPLACE_") for s in cameras)
    ):
        raise ValueError("Specify two distinct camera serials: wrist then third view.")
    if (
        env.keyboard_reward_wrapper != "pnp_human"
        or not env.human_feedback_cfg.wait_for_reset_ready
    ):
        raise ValueError("Use human success/failure and the R ready gate.")
    if cfg.runner.num_data_episodes < 1 or env.max_episode_steps < 1:
        raise ValueError("Episode count and horizon must be positive.")
    if env.override_cfg.max_num_steps != env.max_episode_steps:
        raise ValueError("Native and collection horizons must agree.")


def validate_observation(obs) -> None:
    """Check the exact native observations saved before any CNN resizing."""
    main = numpy_copy(obs["main_images"])
    wrist = numpy_copy(obs["extra_view_images"])
    state = numpy_copy(obs["states"])
    if main.shape != (1, 224, 224, 3) or main.dtype != np.uint8:
        raise ValueError("Expected one main RGB uint8 224x224 image.")
    if wrist.shape != (1, 1, 224, 224, 3) or wrist.dtype != np.uint8:
        raise ValueError("Expected one wrist RGB uint8 224x224 extra view.")
    if state.shape != (1, 14) or not np.isfinite(state).all() or state[0, -1] != -1:
        raise ValueError("Expected finite state14 with fixed gripper=-1.")
    if not str(obs["task_descriptions"][0]).strip():
        raise ValueError("Task language must not be empty.")


def export_success(writer, arrays: dict) -> None:
    """Pair o_t with a_t; retain all six rotation/translation action components."""
    actions = arrays["actions"].astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError("Expected finite six-dimensional actions.")
    if np.abs(actions).max() > 1.00001:
        raise ValueError("Actions must be normalized before environment scaling.")
    writer.add_episode(
        images=arrays["obs_main_images"][:-1],
        wrist_images=arrays["obs_extra_view_images"][:-1, 0],
        extra_view_images=None,
        states=arrays["obs_states"][:-1].astype(np.float32),
        actions=actions,
        task=str(arrays["obs_task_descriptions"][0]),
        is_success=True,
        dones=arrays["terminated"] | arrays["truncated"],
    )
    writer.finalize()
