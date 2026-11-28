"""
Test purpose: Verify nvshmem_broadcast behavior.
API name: torch.ops.symm_mem.nvshmem_broadcast
API schema: symm_mem::nvshmem_broadcast(Tensor(a!) input, int root, str group_name) ->
    Tensor(a!)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Broadcast root selection | Check that all receivers get each selected root's data. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Requires the named accelerator kernel and symmetric peer memory. Without them the transfer is
  skipped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_root_selection(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device)
    value = symm_mem.empty(32, dtype=torch.float32, device=device)
    handle = symm_mem.rendezvous(value, group=group)
    for root in (0, 1):
        value.fill_(rank + 1)
        handle.barrier()
        torch.ops.symm_mem.nvshmem_broadcast(value, root, group)
        case.assertEqual(value.tolist(), [root + 1.0] * 32)
        handle.barrier()


class TestNvshmemBroadcast(TwoRankTestCase):
    def test_nvshmem_broadcast_root_selection(self):
        self.run_on_two_ranks(check_root_selection, shmem_ops=("nvshmem_broadcast",))


if __name__ == "__main__":
    run_tests()
