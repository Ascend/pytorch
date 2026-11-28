"""
Test purpose: Verify one_shot_all_reduce_out behavior.
API name: torch.ops.symm_mem.one_shot_all_reduce_out
API schema: symm_mem::one_shot_all_reduce_out(Tensor input, str reduce_op, str
    group_name, Tensor(a!) out) -> Tensor(a!)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Output aliasing | Reduce distinct rank-local inputs and verify the output contents and storage contract. |

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


def check_rank_sum(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device, nvshmem=False)
    source = symm_mem.empty(32, dtype=torch.float32, device=device).fill_(rank + 1)
    handle = symm_mem.rendezvous(source, group=group)
    handle.barrier()
    out = torch.empty_like(source)
    result = torch.ops.symm_mem.one_shot_all_reduce_out(source, "sum", group, out)
    torch.testing.assert_close(result, torch.full_like(source, 3))
    case.assertEqual(result.data_ptr(), out.data_ptr())
    case.assertEqual(source.tolist(), [rank + 1.0] * 32)


class TestOneShotAllReduceOut(TwoRankTestCase):
    def test_one_shot_all_reduce_out_rank_sum(self):
        self.run_on_two_ranks(check_rank_sum, shmem_ops=("one_shot_all_reduce_out",))


if __name__ == "__main__":
    run_tests()
