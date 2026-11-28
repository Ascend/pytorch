"""
Test purpose: Verify determine_local_world_size behavior.
API name: torch.distributed.run.determine_local_world_size
API signature: determine_local_world_size(nproc_per_node)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Worker count and invalid selectors | Check explicit counts, CPU discovery and invalid device selectors. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestDetermineLocalWorldSize(SupplementTestCase):
    def test_determine_local_world_size_process_count(self):
        import torch
        from torch.distributed.run import determine_local_world_size

        for count in (1, 2, 8):
            self.assertEqual(determine_local_world_size(str(count)), count)
        self.assertEqual(determine_local_world_size("cpu"), torch._utils.cpu_count())
        with self.assertRaises(ValueError):
            determine_local_world_size("unknown-device")


if __name__ == "__main__":
    run_tests()
