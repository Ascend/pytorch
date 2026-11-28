"""
Test purpose: Verify reset_iter_state behavior.
API name: torch.distributed.fsdp.FSDPModule.reset_iter_state
API signature: reset_iter_state(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Iteration recovery | Reset after a failed forward and check gradient reduction. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The failure is injected during forward. Recovery from device faults is outside this test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_recover_aborted_forward(case, rank, device):
    import torch
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)

    def interrupt(module, args):
        raise RuntimeError("interrupted forward")

    hook = model.register_forward_pre_hook(interrupt)
    try:
        with case.assertRaisesRegex(RuntimeError, "interrupted forward"):
            model(torch.ones(1, 4, device=device))
    finally:
        hook.remove()
    model.reset_iter_state()
    backward_on_rank(model, rank, device)
    check_reduced_gradient(model)


class TestResetIterState(TwoRankTestCase):
    def test_reset_iter_state_recover_aborted_forward(self):
        self.run_on_two_ranks(check_recover_aborted_forward)


if __name__ == "__main__":
    run_tests()
