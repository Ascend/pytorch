"""
Test purpose: Verify is_initialized behavior.
API name: torch.distributed.distributed_c10d.is_initialized
API signature: is_initialized()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Group lifecycle | Observe initialization and teardown of an isolated default group. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_group_lifecycle(case, rank, device):
    import torch.distributed as dist

    case.assertTrue(dist.distributed_c10d.is_initialized())
    dist.barrier()
    dist.destroy_process_group()
    case.assertFalse(dist.distributed_c10d.is_initialized())


class TestIsInitialized(TwoRankTestCase):
    def test_is_initialized_group_lifecycle(self):
        self.run_on_two_ranks(check_group_lifecycle)


if __name__ == "__main__":
    run_tests()
