"""
Test purpose: Verify set_separate_reduce_scatter_group behavior.
API name: torch.distributed.fsdp.FSDPModule.set_separate_reduce_scatter_group
API signature: set_separate_reduce_scatter_group(self, enable=True, *, recurse=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Communicator selection | Check gradients with separate and shared groups. |

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


def check_communicator_selection(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    (group,) = model._get_fsdp_state()._fsdp_param_groups
    info = group.mesh_info
    model.set_separate_reduce_scatter_group()
    case.assertIsNotNone(info.reduce_scatter_process_group)
    case.assertIsNot(info.reduce_scatter_process_group, info.shard_process_group)
    backward_on_rank(model, rank, device)
    check_reduced_gradient(model)
    model.set_separate_reduce_scatter_group(False)
    case.assertIsNone(info.reduce_scatter_process_group)
    backward_on_rank(model, rank, device)
    check_reduced_gradient(model)


class TestSetSeparateReduceScatterGroup(TwoRankTestCase):
    def test_set_separate_reduce_scatter_group_communicator_selection(self):
        self.run_on_two_ranks(check_communicator_selection)


if __name__ == "__main__":
    run_tests()
