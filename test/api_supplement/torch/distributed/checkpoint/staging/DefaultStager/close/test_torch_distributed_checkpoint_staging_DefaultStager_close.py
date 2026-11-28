"""
Test purpose: Verify close behavior.
API name: torch.distributed.checkpoint.staging.DefaultStager.close
API signature: close(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Cleanup and pending work | Closing an asynchronous stager waits for its last snapshot and tolerates a second close. |

Execution: Single-process unittest.
Scope and prerequisites:
- The shutdown contract is observed through a pending future, without measuring allocator memory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestClose(SupplementTestCase):
    def test_close_finish_pending_work(self):
        import torch
        from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

        stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=False, use_shared_memory=False, use_non_blocking_copy=False
            )
        )
        try:
            future = stager.stage({"counter": torch.tensor(7)})
        finally:
            stager.close()
        self.assertTrue(future.done())
        self.assertEqual(future.result()["counter"].item(), 7)
        self.assertIsNone(stager.close())


if __name__ == "__main__":
    run_tests()
