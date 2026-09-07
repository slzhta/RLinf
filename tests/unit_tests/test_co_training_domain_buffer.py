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

"""Exercise actor buffer methods without importing the GPU/Ray worker stack."""

from __future__ import annotations

import ast
import asyncio
import math
import time
import unittest
from collections import deque
from pathlib import Path


def _load_buffer_methods() -> type:
    path = Path(__file__).resolve().parents[2] / "rlinf/workers/actor/fsdp_actor_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedFSDPActor"
    )
    names = {
        "_configure_domain_buffer_mode",
        "_append_domain_buffer_blocks",
        "_resolve_domain_train_step_window",
        "_default_max_domain_buffer_steps",
        "_recv_domain_buffered_batch",
        "_domain_buffers_ready_for_train",
        "_select_train_real_steps_from_buffers",
        "_next_receivable_domain_rank",
        "_domain_buffer_step_need",
        "_next_queued_domain_rank",
        "_pop_buffer_blocks",
    }
    methods = [node for node in original.body if getattr(node, "name", "") in names]
    assert len(methods) == len(names), (
        "Actor buffer methods changed; update test harness."
    )
    cls = ast.ClassDef(
        name="BufferActor", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    ast.fix_missing_locations(module)
    namespace = {"asyncio": asyncio, "math": math, "time": time}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["BufferActor"]


BufferActor = _load_buffer_methods()


class Config(dict):
    __getattr__ = dict.__getitem__


def _blocks(domain: str, start: int, count: int) -> list[dict]:
    return [
        {"id": (domain, i), "actions": [i, i + 1], "values": [i, i + 1, i + 2]}
        for i in range(start, start + count)
    ]


def _actor(mode: str = "latest"):
    actor = BufferActor()
    actor._train_batch_steps = 36
    actor._min_train_real_steps = 2
    actor._target_train_real_steps = 4
    actor._max_train_real_steps = 5
    actor._min_train_sim_steps = 31
    actor._target_train_sim_steps = 32
    actor._max_train_sim_steps = 34
    actor._real_receive_steps = 1
    actor._sim_receive_steps = 32
    actor._max_real_buffer_steps = 37
    actor._max_sim_buffer_steps = 68
    actor._actor_train_batch_steps = 240
    actor._rollout_steps_per_trajectory = 120
    actor._real_env_interaction_steps = 0
    actor._sim_env_interaction_steps = 0
    actor._domain_buffer_dropped_rollouts = {"real": 0, "sim": 0}
    actor._real_batch_buffer = []
    actor._sim_batch_buffer = []
    actor._default_real_step_ratio = lambda: 1 / 33
    actor._real_env_ranks = lambda: [0]
    actor._sim_env_ranks = lambda: [1]
    actor._process_domain_trajectory = lambda blocks: blocks
    actor._concat_rollout_batches_along_batch_dim = lambda blocks: blocks
    actor._configure_domain_buffer_mode({"mode": mode})
    actor.queues = {0: deque(), 1: deque()}
    actor.calls = []
    actor.on_receive = lambda rank: None
    actor._keyed_actor_trajectory_qsize = lambda channel, rank: len(actor.queues[rank])

    async def receive(channel, rank):
        actor.calls.append(rank)
        blocks = actor.queues[rank].popleft()
        actor.on_receive(rank)
        return blocks

    actor._recv_keyed_actor_trajectory = receive
    return actor


class DomainBufferTests(unittest.TestCase):
    def test_ratio_window_and_batch_size(self):
        actor = _actor()
        config = Config(
            sample_rollout_length=36,
            real_ratio_min=0.05,
            real_ratio_target=0.11,
            real_ratio_max=0.15,
        )
        self.assertEqual(actor._resolve_domain_train_step_window(config), (36, 2, 4, 5))
        self.assertEqual(36 * 120 // 240, 18)

    def test_dynamic_counts_and_insufficient_data(self):
        for real, sim, expected in [
            (2, 34, 2), (3, 33, 3), (4, 32, 4),
            (5, 31, 5), (5, 34, 4), (2, 32, None),
        ]:
            with self.subTest(real=real, sim=sim):
                actor = _actor()
                actor._real_batch_buffer = _blocks("real", 0, real)
                actor._sim_batch_buffer = _blocks("sim", 0, sim)
                self.assertEqual(
                    actor._select_train_real_steps_from_buffers(), expected
                )

    def test_default_fifo_and_invalid_mode(self):
        actor = _actor()
        actor._configure_domain_buffer_mode({})
        self.assertEqual(actor._domain_buffer_mode, "fifo")
        with self.assertRaisesRegex(ValueError, "fifo or latest"):
            actor._configure_domain_buffer_mode({"mode": "typo"})

    def test_latest_capacity_covers_entire_ratio_window(self):
        for domain, capacity in [("real", 4), ("sim", 32)]:
            with self.subTest(domain=domain):
                actor = _actor()
                setattr(actor, f"_max_{domain}_buffer_steps", capacity)
                with self.assertRaisesRegex(ValueError, "full ratio window"):
                    actor._configure_domain_buffer_mode({"mode": "latest"})
                actor._configure_domain_buffer_mode({"mode": "fifo"})

    def test_capacity_defaults(self):
        actor = _actor()
        self.assertEqual(actor._default_max_domain_buffer_steps(1), 37)
        self.assertEqual(actor._default_max_domain_buffer_steps(32), 68)

    def test_append_trims_only_oldest_blocks(self):
        actor = _actor()
        actor._max_sim_buffer_steps = 34
        actor._sim_batch_buffer = _blocks("sim", 0, 32)
        new = _blocks("sim", 32, 32)
        self.assertEqual(actor._append_domain_buffer_blocks("sim", new), 30)
        self.assertEqual(
            [b["id"][1] for b in actor._sim_batch_buffer], list(range(30, 64))
        )
        self.assertIs(actor._sim_batch_buffer[-1], new[-1])
        self.assertEqual(actor._domain_buffer_dropped_rollouts["sim"], 30)

    def test_fifo_append_does_not_trim(self):
        actor = _actor("fifo")
        actor._max_sim_buffer_steps = 34
        actor._sim_batch_buffer = _blocks("sim", 0, 32)
        self.assertEqual(
            actor._append_domain_buffer_blocks("sim", _blocks("sim", 32, 32)), 0
        )
        self.assertEqual(len(actor._sim_batch_buffer), 64)

    def test_pop_keeps_time_order_and_unused_blocks(self):
        for latest, indices in [(True, [3, 4]), (False, [0, 1])]:
            blocks = _blocks("sim", 0, 5)
            original = blocks.copy()
            selected = BufferActor._pop_buffer_blocks(blocks, 2, latest=latest)
            self.assertEqual([b["id"][1] for b in selected], indices)
            self.assertEqual(len(blocks), 3)
            for block, index in zip(selected, indices):
                self.assertIs(block, original[index])
                self.assertEqual(block["actions"], [index, index + 1])
                self.assertEqual(block["values"], [index, index + 1, index + 2])

    def test_zero_and_invalid_pop(self):
        for latest in (False, True):
            blocks = _blocks("real", 0, 2)
            self.assertEqual(BufferActor._pop_buffer_blocks(blocks, 0, latest), [])
            self.assertEqual(len(blocks), 2)
            for count in (-1, 3):
                with self.assertRaises(ValueError):
                    BufferActor._pop_buffer_blocks(blocks, count, latest)

    def test_latest_receives_even_when_full(self):
        actor = _actor()
        actor._sim_batch_buffer = _blocks("sim", 0, 68)
        actor.queues[1].append(_blocks("sim", 68, 32))
        self.assertEqual(actor._next_receivable_domain_rank(None), (1, "sim"))
        actor._configure_domain_buffer_mode({"mode": "fifo"})
        self.assertIsNone(actor._next_receivable_domain_rank(None))


class ReceiveTests(unittest.IsolatedAsyncioTestCase):
    async def _receive(self, actor):
        return await asyncio.wait_for(
            actor._recv_domain_buffered_batch(None), timeout=2
        )

    async def test_ready_on_entry_drains_but_fifo_does_not(self):
        for mode in ("latest", "fifo"):
            actor = _actor(mode)
            actor._real_batch_buffer = _blocks("real", 0, 4)
            actor._sim_batch_buffer = _blocks("sim", 0, 32)
            actor.queues[0].append(_blocks("real", 4, 1))
            actor.queues[1].append(_blocks("sim", 32, 32))
            selected = await self._receive(actor)
            if mode == "latest":
                self.assertEqual(actor.calls, [0, 1])
                self.assertEqual([b["id"][1] for b in selected[:4]], [1, 2, 3, 4])
                self.assertEqual(
                    [b["id"][1] for b in selected[4:]], list(range(32, 64))
                )
                self.assertEqual(len(actor._sim_batch_buffer), 32)
            else:
                self.assertEqual(actor.calls, [])
                self.assertEqual(selected[-1]["id"], ("sim", 31))

    async def test_minimum_ratio_does_not_wait_for_target(self):
        actor = _actor()
        actor._sim_batch_buffer = _blocks("sim", 0, 34)
        actor.queues[0].extend([_blocks("real", 0, 1), _blocks("real", 1, 1)])
        selected = await self._receive(actor)
        self.assertEqual(len(selected), 36)
        self.assertEqual(
            actor._co_training_buffer_metrics["buffer/train_real_rollouts"], 2
        )

    async def test_drain_snapshot_has_finite_budget(self):
        actor = _actor()
        actor._real_batch_buffer = _blocks("real", 0, 4)
        actor._sim_batch_buffer = _blocks("sim", 0, 32)
        actor.queues[1].append(_blocks("sim", 32, 32))
        actor.on_receive = lambda rank: actor.queues[rank].append(
            _blocks("sim", 64, 32)
        )
        selected = await self._receive(actor)
        self.assertEqual(actor.calls, [1])
        self.assertEqual(len(actor.queues[1]), 1)
        self.assertEqual(selected[-1]["id"], ("sim", 63))

    async def test_full_sim_buffer_keeps_receiving_while_waiting_for_real(self):
        actor = _actor()
        actor._sim_batch_buffer = _blocks("sim", 0, 68)
        actor.queues[0].append(_blocks("real", 0, 1))
        actor.queues[1].extend(_blocks("sim", i, 32) for i in (68, 100, 132))

        def on_receive(rank):
            if rank == 1 and not actor.queues[1]:
                actor.queues[0].append(_blocks("real", 1, 1))

        actor.on_receive = on_receive
        selected = await self._receive(actor)
        self.assertEqual(actor.calls, [0, 1, 1, 1, 0])
        self.assertEqual(len(selected), 36)
        metrics = actor._co_training_buffer_metrics
        self.assertEqual(metrics["buffer/dropped_sim_rollouts"], 96)
        self.assertEqual(metrics["buffer/sim_rollouts_after_recv"], 68)
        self.assertEqual(metrics["buffer/train_sim_rollouts"], 34)
        for domain in ("real", "sim"):
            self.assertEqual(
                metrics[f"buffer/{domain}_rollouts_before_recv"]
                + metrics[f"buffer/received_{domain}_rollouts"]
                - metrics[f"buffer/dropped_{domain}_rollouts"]
                - metrics[f"buffer/train_{domain}_rollouts"],
                metrics[f"buffer/{domain}_rollouts_after_pop"],
            )

    async def test_unused_blocks_are_not_reused_after_consumption(self):
        actor = _actor()
        actor._real_batch_buffer = _blocks("real", 0, 8)
        actor._sim_batch_buffer = _blocks("sim", 0, 64)
        first = await self._receive(actor)
        second = await self._receive(actor)
        self.assertTrue(set(b["id"] for b in first).isdisjoint(b["id"] for b in second))
        self.assertEqual(actor._real_batch_buffer, [])
        self.assertEqual(actor._sim_batch_buffer, [])


if __name__ == "__main__":
    unittest.main()
