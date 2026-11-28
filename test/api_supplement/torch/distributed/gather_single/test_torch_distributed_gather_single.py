"""
Test purpose: Verify gather_single behavior.
API name: torch.distributed.gather_single
API signature: gather_single(tensor, gather_tensor=None, dst=None, group=None, async_op=False,
    group_dst=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Invalid argument | Reject non-tensor input before dispatch. |
| Root selection and async completion | Gather distinct rank values at each root in sync and async modes. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- A positive transfer requires backend gather_single support. Argument validation runs
  independently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_destination_and_async(case, rank, device):
    import torch
    import torch.distributed as dist

    for dst, asynchronous in ((0, False), (1, True)):
        value = torch.tensor([rank, rank + 2], device=device)
        result = value.new_empty(4) if rank == dst else None
        try:
            work = dist.gather_single(value, result, dst=dst, async_op=asynchronous)
        except RuntimeError as error:
            if (
                device.type == "npu"
                and str(error).splitlines()[0] == "Backend hccl does not support gather_single"
            ):
                case.skipTest("The installed HCCL backend does not implement gather_single")
            raise
        if asynchronous:
            work.wait()
        if rank == dst:
            case.assertEqual(result.tolist(), [0, 2, 1, 3])


class TestGatherSingle(TwoRankTestCase):
    def test_gather_single_invalid_input(self):
        import torch.distributed as dist

        with self.assertRaisesRegex(TypeError, "tensor"):
            dist.gather_single([1], dst=0)

    def test_gather_single_destination_and_async(self):
        self.run_on_two_ranks(check_destination_and_async)


if __name__ == "__main__":
    run_tests()
