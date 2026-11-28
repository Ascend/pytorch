"""
Test purpose: Verify shrink_group behavior.
API name: torch.distributed.distributed_c10d.shrink_group
API signature: shrink_group(ranks_to_exclude, group=None, shrink_flags=0, pg_options=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Exclusions and capability | Reject empty exclusions; check surviving membership or the backend capability error. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The two-rank group leaves one surviving member. Shrinking depends on backend support.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_exclusion_and_backend(case, rank, device):
    import torch.distributed as dist

    with case.assertRaisesRegex(ValueError, "empty"):
        dist.shrink_group([])
    group = dist.new_group([0, 1])
    try:
        if rank == 0:
            if group._get_backend(device).supports_shrinking:
                smaller = dist.shrink_group([1], group=group)
                try:
                    case.assertEqual(dist.get_world_size(smaller), 1)
                    case.assertEqual(dist.get_rank(smaller), 0)
                finally:
                    dist.destroy_process_group(smaller)
            else:
                with case.assertRaisesRegex(TypeError, "support shrinking"):
                    dist.shrink_group([1], group=group)
        dist.barrier()
    finally:
        dist.destroy_process_group(group)


class TestShrinkGroup(TwoRankTestCase):
    def test_shrink_group_exclusion_and_backend(self):
        self.run_on_two_ranks(check_exclusion_and_backend)


if __name__ == "__main__":
    run_tests()
