"""
Test purpose: Verify set_manual_backward_finalization behavior.
API name: torch.distributed.fsdp.FSDPModule.set_manual_backward_finalization
API signature: set_manual_backward_finalization(self, enabled)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Manual and automatic backward | Finish a manual backward, then run automatic backward. |

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


def check_manual_mode_lifecycle(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    model.set_manual_backward_finalization(True)
    backward_on_rank(model, rank, device)
    model.finalize_backward()
    check_reduced_gradient(model)
    model.set_manual_backward_finalization(False)
    backward_on_rank(model, rank, device)
    check_reduced_gradient(model)


class TestSetManualBackwardFinalization(TwoRankTestCase):
    def test_set_manual_backward_finalization_manual_mode_lifecycle(self):
        self.run_on_two_ranks(check_manual_mode_lifecycle)


if __name__ == "__main__":
    run_tests()
