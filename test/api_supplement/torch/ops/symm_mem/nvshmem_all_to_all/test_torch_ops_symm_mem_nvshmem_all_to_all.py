"""
Test purpose: Verify nvshmem_all_to_all behavior.
API name: torch.ops.symm_mem.nvshmem_all_to_all
API schema: symm_mem::nvshmem_all_to_all(Tensor input, Tensor(a!) out, str group_name)
    -> Tensor(a!)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Peer block ordering | Exchange one contiguous block per peer and verify both origin and destination indexing. |

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


def check_rank_order(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device)
    source = symm_mem.empty(8, dtype=torch.float32, device=device)
    source.copy_(torch.arange(8, device=device) + rank * 10)
    out = symm_mem.empty(8, dtype=source.dtype, device=device).fill_(-1)
    symm_mem.rendezvous(source, group=group)
    handle = symm_mem.rendezvous(out, group=group)
    handle.barrier()
    torch.ops.symm_mem.nvshmem_all_to_all(source, out, group)
    start = rank * 4
    case.assertEqual(
        out.tolist(), [origin * 10 + start + offset for origin in range(2) for offset in range(4)]
    )


class TestNvshmemAllToAll(TwoRankTestCase):
    def test_nvshmem_all_to_all_rank_order(self):
        self.run_on_two_ranks(check_rank_order, shmem_ops=("nvshmem_all_to_all",))


if __name__ == "__main__":
    run_tests()
