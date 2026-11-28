"""
Test purpose: Verify DefaultStager behavior.
API name: torch.distributed.checkpoint.staging.DefaultStager
API signature: DefaultStager(config=StagingOptions(use_pinned_memory=True, use_shared_memory=True,
    use_async_staging=True, use_non_blocking_copy=True))

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Sync and async staging options | Verify synchronous and asynchronous staging using explicit portable memory options. |

Execution: Single-process unittest.
Scope and prerequisites:
- This case isolates execution mode; pinned and shared allocator behavior is outside the fixture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestDefaultStager(SupplementTestCase):
    def test_default_stager_staging_modes(self):
        from concurrent.futures import Future

        import torch
        from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

        for asynchronous in (False, True):
            stager = DefaultStager(
                StagingOptions(
                    use_async_staging=asynchronous,
                    use_pinned_memory=False,
                    use_shared_memory=False,
                    use_non_blocking_copy=False,
                )
            )
            try:
                result = stager.stage({"x": torch.arange(3)})
                self.assertIs(isinstance(result, Future), asynchronous)
                staged = result.result(timeout=30) if asynchronous else result
                self.assertEqual(staged["x"].tolist(), [0, 1, 2])
            finally:
                stager.close()


if __name__ == "__main__":
    run_tests()
