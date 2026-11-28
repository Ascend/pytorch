"""
Test purpose: Verify set_gradient_divide_factor behavior.
API name: torch.distributed.fsdp.FSDPModule.set_gradient_divide_factor
API signature: set_gradient_divide_factor(self, factor)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Gradient scaling | Check sharded gradients against the scaled rank sum. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Rank-local gradients have an exact closed form. The case uses positive finite factors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_scaled_gradient(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    for factor in (1.0, 2.0, 4.0):
        model.set_gradient_divide_factor(factor)
        backward_on_rank(model, rank, device)
        check_reduced_gradient(model, expected=3.0 / factor)


class TestSetGradientDivideFactor(TwoRankTestCase):
    def test_set_gradient_divide_factor_scaled_gradient(self):
        self.run_on_two_ranks(check_scaled_gradient)


if __name__ == "__main__":
    run_tests()
