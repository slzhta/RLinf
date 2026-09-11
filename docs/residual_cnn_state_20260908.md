# Residual CNN state input on a4

The a4 residual run config now sets actor.model.use_state=true and state_dim=14.
Both train/eval residual environments inherit the switch. The 14 dimensions are
seven joint positions, end-effector xyz, end-effector sxyz Euler angles, and
continuous normalized gripper opening in [-1, 1]. No privileged object state is
included. Existing state_mean/state_std remain empty; no unrelated statistics
or binary-gripper conversion are introduced.

The CNN numeric input has 94 dimensions: 70 base-plan values, 10 mask values,
and 14 robot-state values. Actor and value branches use this input, which is
stored with each PPO sample for likelihood/value replay.

The frozen pi05 nostate checkpoint and discrete_state_input=false are unchanged.
Its preprocessing API still requires a state-shaped field, but pi05 has no
continuous state projection and tokenization does not condition on state.
With fixed images, prompt and RNG seeds, changing all state values by +1 gave
exactly identical OpenPI actions in a real-checkpoint smoke test on GPU 3.
The same state change altered CNN features; value gradients reached the state
channels, and rollout/update logprobs and values matched.

No other training hyperparameters were changed. This changes the CNN input
layer from 80 to 94 dimensions. Start a fresh residual experiment; previous
state-free residual checkpoints cannot be directly resumed into this shape.
The running experiment is unaffected by editing its source YAML.

Regression: tests/unit_tests/test_residual_policy.py now checks PPO updates with
both use_state=False and use_state=True.
