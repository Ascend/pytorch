"""
Test purpose: Verify nvshmem_get_out behavior.
API name: torch.ops.symm_mem.nvshmem_get_out
API schema: symm_mem::nvshmem_get_out(Tensor(a!) dst,
    __torch__.torch.classes.c10d.SymmetricMemory hdl, int offset, int size, int peer) -> ()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Remote subrange and output bounds | Read a peer subrange and check the trailing sentinel. |

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


def check_offset_and_length(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device)
    source = symm_mem.empty(8, dtype=torch.float32, device=device)
    source.copy_(torch.arange(8, device=device) + rank * 10)
    handle = symm_mem.rendezvous(source, group=group)
    handle.barrier()
    out = torch.full((4,), -1.0, device=device)
    torch.ops.symm_mem.nvshmem_get_out(out, handle.boxed(), 2, 3, 1 - rank)
    base = (1 - rank) * 10
    case.assertEqual(out.tolist(), [base + 2, base + 3, base + 4, -1])
    handle.barrier()


class TestNvshmemGetOut(TwoRankTestCase):
    def test_nvshmem_get_out_offset_and_length(self):
        self.run_on_two_ranks(check_offset_and_length, shmem_ops=("nvshmem_get_out",))


if __name__ == "__main__":
    run_tests()
