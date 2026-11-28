"""
Test purpose: Verify set_keep_unsharded_storage behavior.
API name: torch.distributed.fsdp.FSDPModule.set_keep_unsharded_storage
API signature: set_keep_unsharded_storage(self, keep, *, recurse=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Storage reuse | Check storage pointer reuse after reshard and unshard. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_storage_reuse(case, rank, device):
    import torch
    from distributed_test_support import (
        sharded_linear,
    )

    model = sharded_linear(device)
    model.set_keep_unsharded_storage(True)
    model.unshard()
    pointer = model.weight.data_ptr()
    model.reshard()
    model.unshard()
    case.assertEqual(model.weight.data_ptr(), pointer)
    torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
    model.set_keep_unsharded_storage(False)
    model.reshard()


class TestSetKeepUnshardedStorage(TwoRankTestCase):
    def test_set_keep_unsharded_storage_storage_reuse(self):
        self.run_on_two_ranks(check_storage_reuse)


if __name__ == "__main__":
    run_tests()
