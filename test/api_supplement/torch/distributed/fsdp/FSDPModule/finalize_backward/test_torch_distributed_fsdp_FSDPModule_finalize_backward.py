"""
Test purpose: Verify finalize_backward behavior.
API name: torch.distributed.fsdp.FSDPModule.finalize_backward
API signature: finalize_backward(self, *, async_op=False)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Manual backward finalization | Check the pre-forward no-op and the final reduced gradient. |

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


def check_manual_finalization(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    model.set_manual_backward_finalization(True)
    case.assertIsNone(model.finalize_backward())
    backward_on_rank(model, rank, device)
    case.assertIsNone(model.finalize_backward())
    check_reduced_gradient(model)


class TestFinalizeBackward(TwoRankTestCase):
    def test_finalize_backward_manual_finalization(self):
        self.run_on_two_ranks(check_manual_finalization)


if __name__ == "__main__":
    run_tests()
