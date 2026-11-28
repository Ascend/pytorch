"""
Test purpose: Verify _reconfigure behavior.
API name: torch.distributed.distributed_c10d._reconfigure
API signature: _reconfigure(uuid, handles, group=None, timeout=None, hints=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Backend reconfiguration | Exchange peer handles or check the unsupported-backend error. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Membership changes require support for reconfiguration. Timeout and backend-specific hints are
  optional.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_backend_reconfiguration(case, rank, device):
    import torch
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    group = dist.group.WORLD
    if not c10d._supports_reconfigure(group):
        with case.assertRaisesRegex(RuntimeError, "support|implement"):
            c10d._reconfigure(1, [], group=group)
        return
    handles = [None, None]
    dist.all_gather_object(handles, c10d._get_reconfigure_handle(group), group=group)
    work = c10d._reconfigure(1, handles, group=group)
    work.wait()
    value = torch.tensor([rank + 1], device=device)
    dist.all_reduce(value, group=group)
    case.assertEqual(value.item(), 3)


class TestReconfigure(TwoRankTestCase):
    def test_reconfigure_backend_reconfiguration(self):
        self.run_on_two_ranks(check_backend_reconfiguration)


if __name__ == "__main__":
    run_tests()
