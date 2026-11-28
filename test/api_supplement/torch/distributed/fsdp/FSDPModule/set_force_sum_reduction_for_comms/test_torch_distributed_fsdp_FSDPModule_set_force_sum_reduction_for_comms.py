"""
Test purpose: Verify set_force_sum_reduction_for_comms behavior.
API name: torch.distributed.fsdp.FSDPModule.set_force_sum_reduction_for_comms
API signature: set_force_sum_reduction_for_comms(self, enable)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Reduction policy | Check the policy and gradients with forced SUM on and off. |

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


def check_reduction_policy(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    (group,) = model._get_fsdp_state()._fsdp_param_groups
    for enabled in (True, False):
        model.set_force_sum_reduction_for_comms(enabled)
        case.assertIs(group.force_sum_reduction_for_comms, enabled)
        backward_on_rank(model, rank, device)
        check_reduced_gradient(model)


class TestSetForceSumReductionForComms(TwoRankTestCase):
    def test_set_force_sum_reduction_for_comms_reduction_policy(self):
        self.run_on_two_ranks(check_reduction_policy)


if __name__ == "__main__":
    run_tests()
