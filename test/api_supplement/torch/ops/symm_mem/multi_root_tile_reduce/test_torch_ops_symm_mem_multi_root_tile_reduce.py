"""
Test purpose: Verify multi_root_tile_reduce behavior.
API name: torch.ops.symm_mem.multi_root_tile_reduce
API schema: symm_mem::multi_root_tile_reduce(Tensor[] in_tiles, Tensor(a!) out_tile,
    int[] roots, str group_name, str reduce_op="sum") -> ()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Root assignments and output regions | Check root tile sums and untouched output regions. |

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
    tiles = [source[:, :32], source[:, 32:]]
    local_slice = slice(rank * 32, (rank + 1) * 32)
    torch.ops.symm_mem.multi_root_tile_reduce(tiles, out[:, local_slice], [0, 1], group)
    expected[:, local_slice].fill_(3)
    torch.testing.assert_close(out, expected)


class TestMultiRootTileReduce(TwoRankTestCase):
    def test_multi_root_tile_reduce_root_tiles(self):
        self.run_on_two_ranks(check_root_tiles, shmem_ops=("multi_root_tile_reduce",))


if __name__ == "__main__":
    run_tests()
