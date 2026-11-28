"""
Test purpose: Verify set_allocate_memory_from_process_group_for_comm behavior.
API name: torch.distributed.fsdp.FSDPModule.set_allocate_memory_from_process_group_for_comm
API signature: set_allocate_memory_from_process_group_for_comm(self, enable)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Buffer allocator selection | Check gradients with either allocator. |

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


def check_allocator_selection(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    (group,) = model._get_fsdp_state()._fsdp_param_groups
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        DefaultAllGather,
        ProcessGroupAllocAllGather,
    )

    for enabled, allocator in ((True, ProcessGroupAllocAllGather), (False, DefaultAllGather)):
        model.set_allocate_memory_from_process_group_for_comm(enabled)
        case.assertIsInstance(group._all_gather_comm, allocator)
        backward_on_rank(model, rank, device)
        check_reduced_gradient(model)


class TestSetAllocateMemoryFromProcessGroupForComm(TwoRankTestCase):
    def test_set_allocate_memory_from_process_group_for_comm_allocator_selection(self):
        self.run_on_two_ranks(check_allocator_selection)


if __name__ == "__main__":
    run_tests()
