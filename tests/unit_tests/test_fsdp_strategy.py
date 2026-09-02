from types import SimpleNamespace

import torch.nn as nn

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy


def test_before_micro_batch_supports_unwrapped_single_process_model() -> None:
    strategy = object.__new__(FSDPStrategy)
    strategy.cfg = SimpleNamespace(
        fsdp_config=SimpleNamespace(enable_gradient_accumulation=True)
    )

    with strategy.before_micro_batch(nn.Linear(2, 2), is_last_micro_batch=False):
        pass
