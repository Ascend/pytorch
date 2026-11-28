"""
Test purpose: Verify GradientReductionHandle behavior.
API name: torch.distributed.fsdp.GradientReductionHandle
API signature: GradientReductionHandle()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Async return type and gradients | Check the real async return type and use its gradients after waiting. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_asynchronous_handle(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    from torch.distributed.fsdp import GradientReductionHandle

    model.set_manual_backward_finalization(True)
    backward_on_rank(model, rank, device)
    handle = model.finalize_backward(async_op=True)
    case.assertIsInstance(handle, GradientReductionHandle)
    handle.wait()
    check_reduced_gradient(model)


class TestGradientReductionHandle(TwoRankTestCase):
    def test_gradient_reduction_handle_asynchronous_handle(self):
        self.run_on_two_ranks(check_asynchronous_handle)


if __name__ == "__main__":
    run_tests()
