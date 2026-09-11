# Residual entropy shape fix

ResidualPolicy now returns entropy as [B, 1, action_dim]. Chunk aggregation
therefore yields [B, 1], matching the actor loss mask instead of broadcasting
[B] with [B, 1] to [B, B]. Log probabilities and executed actions are unchanged.
The fix is local to residual policy; shared CNN/OpenPI behavior is unchanged.

CPU regression exercises the actual residual methods and entropy utilities
using a lightweight Gaussian parent, without vision/GPU dependencies. It checks
action/chunk aggregation, inactive gripper, full/partial/empty masks and equal
accumulated entropy gradients with micro batches 20 and 120. It does not replace
a full GPU PPO integration test.

Run `.venv/bin/python tests/unit_tests/test_residual_entropy_shape.py`.
At initial logstd=0, six active dimensions have entropy about 8.5136 independent
of micro batch. Existing processes retain loaded Python code: relaunch training
to use this fix. For a clean comparison, start fresh rather than resume a run
trained with the inflated entropy term. The existing total_loss metric is still
logged after gradient-accumulation scaling; do not compare its raw magnitude
across micro-batch configurations.
