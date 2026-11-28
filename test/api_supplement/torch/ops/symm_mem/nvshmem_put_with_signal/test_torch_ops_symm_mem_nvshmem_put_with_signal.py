"""
Test purpose: Verify nvshmem_put_with_signal behavior.
API name: torch.ops.symm_mem.nvshmem_put_with_signal
API schema: symm_mem::nvshmem_put_with_signal(Tensor(a) tensor, Tensor(a) sigpad, int
    signal, int peer) -> ()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Signaled remote writes | Check that a signaled write delivers the full peer payload. |

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


def check_signaled_peer_write(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device)
    source_rank = 0
    destination_rank = 1 - source_rank
    value = symm_mem.empty(32, dtype=torch.float32, device=device).fill_(rank + 1)
    handle = symm_mem.rendezvous(value, group=group)
    signal = handle.get_signal_pad(rank)
    # Channel zero overlaps the signal pad used by these operations.
    handle.barrier(channel=1)
    if rank == source_rank:
        torch.ops.symm_mem.nvshmem_put_with_signal(value, signal, 5, destination_rank)
    else:
        torch.ops.symm_mem.nvshmem_wait_for_signal(signal, 5, source_rank)
        case.assertEqual(value.tolist(), [source_rank + 1.0] * 32)
    handle.barrier(channel=1)


class TestNvshmemPutWithSignal(TwoRankTestCase):
    def test_nvshmem_put_with_signal_signaled_peer_write(self):
        self.run_on_two_ranks(
            check_signaled_peer_write,
            shmem_ops=("nvshmem_put_with_signal", "nvshmem_wait_for_signal"),
        )


if __name__ == "__main__":
    run_tests()
