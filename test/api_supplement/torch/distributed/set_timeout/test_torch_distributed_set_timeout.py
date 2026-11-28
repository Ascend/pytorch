"""
Test purpose: Verify set_timeout behavior.
API name: torch.distributed.set_timeout
API signature: set_timeout(timeout, group=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Timeout scope | Set a subgroup timeout, observe it in its options and verify default-group isolation. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The case checks timeout configuration without inducing a collective timeout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_timeout_scope(case, rank, device):
    from datetime import timedelta

    import torch.distributed as dist

    world_timeout = dist.group.WORLD._get_backend(device).options._timeout
    group = dist.new_group([0, 1])
    try:
        duration = timedelta(seconds=73)
        dist.set_timeout(duration, group=group)
        case.assertEqual(group._get_backend(device).options._timeout, duration)
        case.assertEqual(dist.group.WORLD._get_backend(device).options._timeout, world_timeout)
        with case.assertRaises(TypeError):
            dist.set_timeout(73, group=group)
    finally:
        dist.destroy_process_group(group)


class TestSetTimeout(TwoRankTestCase):
    def test_set_timeout_timeout_scope(self):
        self.run_on_two_ranks(check_timeout_scope)


if __name__ == "__main__":
    run_tests()
