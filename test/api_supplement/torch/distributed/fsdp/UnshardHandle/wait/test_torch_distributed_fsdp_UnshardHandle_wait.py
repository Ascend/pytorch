"""
Test purpose: Verify wait behavior.
API name: torch.distributed.fsdp.UnshardHandle.wait
API signature: wait(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Repeated wait | Wait twice on a real async handle and read the gathered parameters across two unshard cycles. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Both waits occur on the same stream after a real asynchronous unshard.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_repeated_wait(case, rank, device):
    import torch
    from distributed_test_support import (
        sharded_linear,
    )

    model = sharded_linear(device)
    for _ in range(2):
        handle = model.unshard(async_op=True)
        case.assertIsNone(handle.wait())
        case.assertIsNone(handle.wait())
        torch.testing.assert_close(model.weight, torch.ones((4, 4), device=device))
        model.reshard()


class TestWait(TwoRankTestCase):
    def test_wait_repeated_wait(self):
        self.run_on_two_ranks(check_repeated_wait)


if __name__ == "__main__":
    run_tests()
