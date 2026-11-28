"""
Test purpose: Verify tile_reduce behavior.
API name: torch.ops.symm_mem.tile_reduce
API schema: symm_mem::tile_reduce(Tensor in_tile, Tensor(a!) out_tile, int root, str
    group_name, str reduce_op="sum") -> ()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Root-only reduction | Check the rank sum at the root and preserve the untouched output region. |

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


def check_root_tiles(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device, multicast=True)
    source = symm_mem.empty((32, 64), dtype=torch.float32, device=device).fill_(rank + 1)
    out = symm_mem.empty((32, 64), dtype=torch.float32, device=device).zero_()
    handle = symm_mem.rendezvous(out, group=group)
    handle.barrier()
    expected = torch.zeros_like(out)
    torch.ops.symm_mem.tile_reduce(source[:, :32], out[:, :32], 0, group)
    if rank == 0:
        expected[:, :32].fill_(3)
    torch.testing.assert_close(out, expected)


class TestTileReduce(TwoRankTestCase):
    def test_tile_reduce_root_tiles(self):
        self.run_on_two_ranks(check_root_tiles, shmem_ops=("tile_reduce",))


if __name__ == "__main__":
    run_tests()
