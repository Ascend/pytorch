"""
Test purpose: Verify UnshardHandle behavior.
API name: torch.distributed.fsdp.UnshardHandle
API signature: UnshardHandle()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Sync and async return types | Check sync and async return types and full parameter shape. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The returned handle is checked on a module with weights; an empty module has no parameters to
  gather.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_async_result_type(case, rank, device):
    from distributed_test_support import (
        sharded_linear,
    )

    model = sharded_linear(device)
    from torch.distributed.fsdp import UnshardHandle

    handle = model.unshard(async_op=True)
    case.assertIsInstance(handle, UnshardHandle)
    handle.wait()
    case.assertEqual(tuple(model.weight.shape), (4, 4))
    model.reshard()
    case.assertIsNone(model.unshard())


class TestUnshardHandle(TwoRankTestCase):
    def test_unshard_handle_async_result_type(self):
        self.run_on_two_ranks(check_async_result_type)


if __name__ == "__main__":
    run_tests()
