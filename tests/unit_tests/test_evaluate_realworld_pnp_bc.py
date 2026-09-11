# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from toolkits.replay_buffer.evaluate_realworld_pnp_bc import _sequence_metrics


def test_sequence_metrics_accepts_small_transition_delay_and_tracks_flicker():
    target = torch.tensor([1, 1, 1, 0, 0, 0, 1, 1, 1] * 2, dtype=torch.bool)
    predicted = torch.tensor(
        [
            1,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            1,
        ],
        dtype=torch.bool,
    )

    metrics = _sequence_metrics(predicted, target, [slice(0, 9), slice(9, 18)])

    assert metrics["strict_open_close_open_episode_rate"] == 0.5
    assert metrics["tolerant_open_close_open_episode_rate"] == 1.0
    assert metrics["mean_predicted_transitions_per_episode"] == 3.0
    assert metrics["tolerant_transition_close_recall"] == 1.0
    assert metrics["tolerant_transition_open_recall"] == 1.0
    assert metrics["mean_transition_close_latency_steps"] == pytest.approx(0.5)
    assert metrics["mean_transition_open_latency_steps"] == 0.0
