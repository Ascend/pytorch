"""
Test purpose: Verify split_group behavior.
API name: torch.distributed.split_group
API signature: split_group(parent_pg=None, split_ranks=None, timeout=None, pg_options=None,
    group_desc=None, backend=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Partition membership | Reject empty partitions; verify singleton subgroup rank mapping on a splitting backend. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_partition_membership(case, rank, device):
    import torch.distributed as dist

    with case.assertRaisesRegex(ValueError, "empty"):
        dist.split_group(split_ranks=[])
    backend = dist.group.WORLD._get_backend(device)
    if not backend.supports_splitting:
        with case.assertRaisesRegex(RuntimeError, "support splitting"):
            dist.split_group(split_ranks=[[0], [1]])
        return
    group = dist.split_group(split_ranks=[[0], [1]])
    try:
        case.assertEqual(dist.get_process_group_ranks(group), [rank])
        case.assertEqual(dist.get_rank(group), 0)
    finally:
        dist.destroy_process_group(group)


class TestSplitGroup(TwoRankTestCase):
    def test_split_group_partition_membership(self):
        self.run_on_two_ranks(check_partition_membership)


if __name__ == "__main__":
    run_tests()
