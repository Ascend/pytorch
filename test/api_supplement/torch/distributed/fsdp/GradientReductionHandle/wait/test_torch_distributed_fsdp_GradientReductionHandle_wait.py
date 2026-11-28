"""
Test purpose: Verify wait behavior.
API name: torch.distributed.fsdp.GradientReductionHandle.wait
API signature: wait(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Wait and invalidation | Waiting is idempotent after success; resetting a pending iteration invalidates its handle. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_repeat_and_invalidate(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    model.set_manual_backward_finalization(True)
    backward_on_rank(model, rank, device)
    handle = model.finalize_backward(async_op=True)
    case.assertIsNone(handle.wait())
    case.assertIsNone(handle.wait())
    check_reduced_gradient(model)
    backward_on_rank(model, rank, device)
    invalidated = model.finalize_backward(async_op=True)
    model.reset_iter_state()
    with case.assertRaisesRegex(RuntimeError, "reset|invalid"):
        invalidated.wait()


class TestWait(TwoRankTestCase):
    def test_wait_repeat_and_invalidate(self):
        self.run_on_two_ranks(check_repeat_and_invalidate)


if __name__ == "__main__":
    run_tests()
