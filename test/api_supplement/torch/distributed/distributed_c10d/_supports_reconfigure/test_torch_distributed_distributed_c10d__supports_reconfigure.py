"""
Test purpose: Verify _supports_reconfigure behavior.
API name: torch.distributed.distributed_c10d._supports_reconfigure
API signature: _supports_reconfigure(group=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Capability and optional group | Default and explicit world-group queries agree with the backend capability. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- This query reports a capability. It does not change group membership.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_group_capability(case, rank, device):
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    expected = dist.group.WORLD.supports_reconfigure
    case.assertIs(c10d._supports_reconfigure(), expected)
    case.assertIs(c10d._supports_reconfigure(group=dist.group.WORLD), expected)


class TestSupportsReconfigure(TwoRankTestCase):
    def test_supports_reconfigure_group_capability(self):
        self.run_on_two_ranks(check_group_capability)


if __name__ == "__main__":
    run_tests()
