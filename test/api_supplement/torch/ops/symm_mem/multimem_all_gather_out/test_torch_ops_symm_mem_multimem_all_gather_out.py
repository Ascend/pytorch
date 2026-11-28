"""
Test purpose: Verify multimem_all_gather_out behavior.
API name: torch.ops.symm_mem.multimem_all_gather_out
API schema: symm_mem::multimem_all_gather_out(Tensor input, str group_name, Tensor(a!)
    out) -> Tensor(a!)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Gathered blocks and output aliasing | Gather one block per rank and check output aliasing. |

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


def check_gather_into_output(case, rank, device):
    import torch
    from distributed_test_support import init_symmetric_memory
    from torch.distributed import _symmetric_memory as symm_mem

    group = init_symmetric_memory(case, device, nvshmem=False, multicast=True)
    source = torch.full((32,), rank + 1.0, device=device)
    out = symm_mem.empty(64, dtype=source.dtype, device=device).zero_()
    handle = symm_mem.rendezvous(out, group=group)
    handle.barrier()
    result = torch.ops.symm_mem.multimem_all_gather_out(source, group, out)
    case.assertEqual(result.data_ptr(), out.data_ptr())
    case.assertEqual(out.tolist(), [1.0] * 32 + [2.0] * 32)


class TestMultimemAllGatherOut(TwoRankTestCase):
    def test_multimem_all_gather_out_gather_into_output(self):
        self.run_on_two_ranks(check_gather_into_output, shmem_ops=("multimem_all_gather_out",))


if __name__ == "__main__":
    run_tests()
