"""
Test purpose: Verify _get_reconfigure_handle behavior.
API name: torch.distributed.distributed_c10d._get_reconfigure_handle
API signature: _get_reconfigure_handle(group=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Reconfiguration handle | Check the backend handle or the unsupported-backend error. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Handle creation depends on the communicator advertising reconfiguration support.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_backend_handle(case, rank, device):
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    for group in (None, dist.group.WORLD):
        if c10d._supports_reconfigure(group):
            handle = c10d._get_reconfigure_handle(group)
            case.assertIsInstance(handle, str)
            case.assertTrue(handle)
        else:
            with case.assertRaisesRegex(RuntimeError, "support|implement"):
                c10d._get_reconfigure_handle(group)


class TestGetReconfigureHandle(TwoRankTestCase):
    def test_get_reconfigure_handle_backend_handle(self):
        self.run_on_two_ranks(check_backend_handle)


if __name__ == "__main__":
    run_tests()
