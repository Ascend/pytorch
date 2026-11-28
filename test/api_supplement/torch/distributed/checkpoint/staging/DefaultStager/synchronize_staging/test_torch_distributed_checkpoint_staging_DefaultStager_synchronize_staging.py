"""
Test purpose: Verify synchronize_staging behavior.
API name: torch.distributed.checkpoint.staging.DefaultStager.synchronize_staging
API signature: synchronize_staging(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Staging synchronization | Synchronize before submitting work and after submitting an asynchronous snapshot. |

Execution: Single-process unittest.
Scope and prerequisites:
- Synchronization is observed through completion. The test makes no timing guarantees.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestSynchronizeStaging(SupplementTestCase):
    def test_synchronize_staging_wait_for_snapshot(self):
        import torch
        from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

        stager = DefaultStager(
            StagingOptions(
                use_pinned_memory=False, use_shared_memory=False, use_non_blocking_copy=False
            )
        )
        try:
            self.assertIsNone(stager.synchronize_staging())
            future = stager.stage({"x": torch.tensor(13)})
            self.assertIsNone(stager.synchronize_staging())
            self.assertTrue(future.done())
            self.assertEqual(future.result()["x"].item(), 13)
        finally:
            stager.close()


if __name__ == "__main__":
    run_tests()
