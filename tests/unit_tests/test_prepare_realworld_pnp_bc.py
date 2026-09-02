# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from toolkits.replay_buffer.prepare_realworld_pnp_bc import (
    prepare_action_targets,
    select_training_window,
    validate_open_close_open,
)


def test_select_training_window_keeps_one_idle_context_step():
    actions = np.zeros((8, 7), dtype=np.float32)
    actions[2:6, 0] = 0.5

    window = select_training_window(actions, context_steps=1)

    assert window == slice(1, 7)


def test_prepare_action_targets_clips_motion_and_persists_gripper_state():
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[0, 2] = 1.2
    actions[1, 6] = -0.9
    actions[3, 6] = 0.95
    states = np.zeros((5, 19), dtype=np.float32)
    states[:2, 0] = 0.08
    states[2:4, 0] = 0.0
    states[4, 0] = 0.08

    targets, transitions = prepare_action_targets(
        actions, states, gripper_width_threshold=0.04
    )

    np.testing.assert_array_equal(targets[:, 6], [1.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(transitions, [False, True, False, True, False])
    assert targets[0, 2] == 1.0
    assert np.all(np.abs(targets) <= 1.0)
    validate_open_close_open(targets, transitions, episode_index=7)
