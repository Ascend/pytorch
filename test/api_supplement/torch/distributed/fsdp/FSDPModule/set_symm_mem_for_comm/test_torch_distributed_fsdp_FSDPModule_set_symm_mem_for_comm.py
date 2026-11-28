"""
Test purpose: Verify set_symm_mem_for_comm behavior.
API name: torch.distributed.fsdp.FSDPModule.set_symm_mem_for_comm
API signature: set_symm_mem_for_comm(self, backend='NCCL')

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Incompatible allocator configuration | Check that a custom gather blocks the allocator change. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The custom-allocator conflict is device-independent. NCCL symmetric-memory execution requires an
  NCCL environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_custom_collective_conflict(case, rank, device):
    from distributed_test_support import (
        sharded_linear,
    )

    model = sharded_linear(device)
    from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather

    class StandaloneGather(AllGather):
        def allocate(self, *args, **kwargs):
            raise AssertionError("configuration must not allocate a tensor")

        def __call__(self, *args, **kwargs):
            raise AssertionError("configuration must not launch a collective")

    model.set_custom_all_gather(StandaloneGather())
    with case.assertRaisesRegex(AssertionError, "custom"):
        model.set_symm_mem_for_comm()


class TestSetSymmMemForComm(TwoRankTestCase):
    def test_set_symm_mem_for_comm_custom_collective_conflict(self):
        self.run_on_two_ranks(check_custom_collective_conflict)


if __name__ == "__main__":
    run_tests()
