Residual PPO with a Frozen OpenPI Base
=====================================

This example adds action-space residual RL to the existing PnP digital twin.
It uses the standard ``train_embodied_agent.py``, ``EmbodiedRunner``,
``EmbodiedFSDPActor``, HF rollout worker and on-policy rollout batches.
The first version supports simulation on one node, with one pipeline stage.
It does not enable real-world or co-training execution.

Code organization
-----------------

* ``rlinf/models/embodiment/residual_policy/``: a CNNPolicy subclass and residual
  configuration validation. Image normalization, ResNet encoders, numeric-state
  projection, Gaussian sampling, entropy and value prediction use CNNPolicy.
* ``rlinf/envs/residual.py``: vectorized base-action cache and action composition.
* ``rlinf/envs/maniskill/residual_maniskill_env.py``: ManiSkill adapter that
  loads the frozen OpenPI base, composes actions and exposes base conditions.
* ``examples/embodiment/config/model/residual_policy.yaml``: inherits ``model/cnn_policy.yaml``
  and adds residual conditioning and exploration settings.
* ``examples/embodiment/config/model/pi0_5.yaml``: existing OpenPI defaults,
  reused under ``base_model`` with PnP overrides in the experiment configuration.
* ``examples/embodiment/config/maniskill_pick_and_place_ppo_residual.yaml``:
  experiment, placement, environment and PPO settings.
* ``examples/embodiment/run_maniskill_residual.sh``: thin launcher for the
  existing training entry point, accepting Hydra overrides.
* ``tests/unit_tests/test_residual_policy.py``: CPU policy, cache, PPO likelihood and
  configuration tests; these require the RLinf Python dependencies.

Only model registration, environment selection, rollout mode dispatch and
the observation transport schema have opt-in additions. Existing SFT
configuration files and OpenPI model implementation are unchanged.

Execution and learning
----------------------

The frozen base is loaded in the GPU environment worker. Training and
evaluation environments in that process share its weights but use separate
action caches. There is no base model in the actor or rollout
policy, so weight synchronization and residual checkpoints contain only
the small policy and its critic. Keep the resolved experiment configuration
and original base checkpoint together when reproducing an experiment.

OpenPI receives the original images and states through its existing input
and output transforms. A ten-step action chunk is cached independently per
environment. Each residual step receives the current images, remaining
base actions and a padding mask. Only exhausted/reset caches call OpenPI.
The residual outputs one action per control step:

.. code-block:: text

   executed = clip(base_action + residual_scale * clip(residual_action, -1, 1), -1, 1)

Composition happens after OpenPI unnormalization and before the existing
normalized delta-pose controller. The default scale is 0.1 on the six arm
dimensions and zero on the gripper. This is a fraction of the controller
input range, not meters/radians. The supplied simulation controller runs at
5 Hz; this is not an assertion of real-world timing equivalence.

The residual inherits the existing CNNPolicy and its 128x128 NHWC image
preprocessing. By default the native pretrained ResNet backbone stays frozen;
its pooling/projection layers, actor and value head are trained. Set
``+actor.model.encoder_config.freeze_backbone=false`` to also train the backbone.
The numeric ``states`` channel contains the flattened remaining base plan and
mask (80 values for a ten-step, seven-dimensional plan). With
``actor.model.use_state=false`` it contains no direct robot proprioception.
The frozen SFT base still receives its original state input, so its actions
may convey state-derived information. Enabling ``actor.model.use_state=true``
appends robot states to this condition. ``state_dim``, ``state_mean`` and
``state_std`` refer only to these optional robot states. Switching this option
changes the model shape and requires a compatible residual checkpoint.

Rollout uses the native CNN schema: images, packed ``states``, sampled
Gaussian ``action``, old log probabilities and old values. PPO recomputes the
native Gaussian likelihood and analytical entropy; inactive action channels
have zero log probability and entropy contributions. Only environment execution
clips and scales the sampled residual. The stored action remains unclipped.
The existing GAE and ``decoupled_actor_critic`` loss perform the update without
a new runner, loss or replay buffer. No OpenPI inference occurs during updates.
Final observations retain base conditions for time-limit value bootstrapping;
true terminal states do not bootstrap. The base cache advances only on actual
environment steps, not on extra value/bootstrap evaluations.


Dependencies and launch
-----------------------

Use the existing Linux environment with RLinf, OpenPI, ManiSkill and the PnP
assets, plus the same ``resnet10_pretrained.pt`` used by the existing CNN
examples. Set ``actor.model.model_path`` to the directory containing this file;
rollout inherits that path. This is separate from the OpenPI checkpoint and
``runner.ckpt_path``. No new dependency, installer or Docker target is introduced.
The default base path is the user's verified 4090 checkpoint; override
``base_model.model_path`` for another machine. Its matching normalization
assets and ``openpi_data.repo_id`` are required.

First run a short simulation smoke test in an appropriate simulation Ray
session. Do not repurpose a Ray session actively controlling a robot.

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual \
     actor.model.model_path=/path/to/resnet_weights \
     runner.max_epochs=2 algorithm.update_epoch=1 runner.save_interval=1

After setting the encoder directory in the experiment configuration, train
with the defaults or enable direct residual state input:

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh
   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual actor.model.use_state=true

Evaluate the base through the new simulation adapter with residual disabled:

.. code-block:: bash

   bash examples/embodiment/run_maniskill_residual.sh \
     maniskill_pick_and_place_ppo_residual \
     runner.only_eval=true actor.model.enabled=false

For residual evaluation, keep ``enabled=true`` and set ``runner.ckpt_path``
to the residual ``full_weights.pt`` file. This must not point at OpenPI SFT
weights. For training continuation use the standard ``runner.resume_dir``.

Validation
----------

.. code-block:: bash

   python -m pytest tests/unit_tests/test_residual_policy.py -q

The tests cover state on/off behavior, the zero initial residual mean,
gripper masking, partial resets, chunk rollover, preserved transport and
rollout fields, native CNN probability/value parity and unclipped-action likelihood recomputation, and an update using
the existing PPO loss. A short GPU training run is
still required to validate Ray/FSDP, simulator assets and the actual base
checkpoint together. Passing these tests does not demonstrate a success-rate
improvement; compare base-only and residual policies on fixed evaluation
seeds after training.

.. note::

   TODO(agent): The initial implementation was written locally. GPU/Ray
   training and checkpoint-backed simulation have not yet been verified.
   Local pytest collection is currently blocked by a Windows PyTorch
   ``c10.dll`` initialization error; no runtime test pass is claimed.
