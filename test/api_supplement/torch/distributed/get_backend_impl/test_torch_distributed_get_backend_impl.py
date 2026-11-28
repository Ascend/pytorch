"""
Test purpose: Verify get_backend_impl behavior.
API name: torch.distributed.get_backend_impl
API signature: get_backend_impl(group=None, device=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Group argument forms | Group object, name and default lookup resolve to the same native backend. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_lookup_by_group_name(case, rank, device):
    import torch.distributed as dist

    world = dist.group.WORLD
    expected = world._get_backend(device)
    for group in (None, world, world.group_name):
        case.assertIs(dist.get_backend_impl(group=group, device=device), expected)


class TestGetBackendImpl(TwoRankTestCase):
    def test_get_backend_impl_lookup_by_group_name(self):
        self.run_on_two_ranks(check_lookup_by_group_name)


if __name__ == "__main__":
    run_tests()
