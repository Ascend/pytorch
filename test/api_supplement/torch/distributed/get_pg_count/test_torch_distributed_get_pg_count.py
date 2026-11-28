"""
Test purpose: Verify get_pg_count behavior.
API name: torch.distributed.get_pg_count
API signature: get_pg_count()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Subgroup creation | Observe the monotonically assigned process-group count when a new group is created. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_subgroup_creation(case, rank, device):
    import torch.distributed as dist

    before = dist.get_pg_count()
    case.assertGreaterEqual(before, 1)
    group = dist.new_group([0, 1])
    try:
        case.assertEqual(dist.get_pg_count(), before + 1)
    finally:
        dist.destroy_process_group(group)


class TestGetPgCount(TwoRankTestCase):
    def test_get_pg_count_subgroup_creation(self):
        self.run_on_two_ranks(check_subgroup_creation)


if __name__ == "__main__":
    run_tests()
