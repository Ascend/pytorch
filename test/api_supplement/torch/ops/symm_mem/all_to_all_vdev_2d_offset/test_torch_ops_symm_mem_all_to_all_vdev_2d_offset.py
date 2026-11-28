"""
Test purpose: Verify all_to_all_vdev_2d_offset behavior.
API name: torch.ops.symm_mem.all_to_all_vdev_2d_offset
API schema: symm_mem::all_to_all_vdev_2d_offset(Tensor input, Tensor(a!) out, Tensor
    in_splits_offsets, Tensor(a!) out_splits_offsets, str group_name) -> ()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Explicit input offsets | Check peer block lengths, exclusive offsets and payloads. |

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


def check_peer_blocks(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device)
    source = symm_mem.empty(8, dtype=torch.float32, device=device)
    source.copy_(torch.arange(8, device=device) + 10 * rank)
    out = symm_mem.empty(8, dtype=torch.float32, device=device).fill_(-1)
    splits = symm_mem.empty((2, 2), dtype=torch.int64, device=device)
    splits.copy_(torch.tensor([[2, 2], [1, 5]], device=device))
    metadata = symm_mem.empty((2, 2), dtype=torch.int64, device=device).fill_(-1)
    handle = symm_mem.rendezvous(out, group=group)
    handle.barrier()
    torch.ops.symm_mem.all_to_all_vdev_2d_offset(source, out, splits, metadata, group)
    case.assertEqual(metadata.tolist(), [[2, 2], [0, 2]])
    start = 1 + 4 * rank
    case.assertEqual(out[:4].tolist(), [start, start + 1, start + 10, start + 11])
    case.assertEqual(out[4:].tolist(), [-1] * 4)


class TestAllToAllVdev2dOffset(TwoRankTestCase):
    def test_all_to_all_vdev_2d_offset_peer_blocks(self):
        self.run_on_two_ranks(check_peer_blocks, shmem_ops=("all_to_all_vdev_2d_offset",))


if __name__ == "__main__":
    run_tests()
