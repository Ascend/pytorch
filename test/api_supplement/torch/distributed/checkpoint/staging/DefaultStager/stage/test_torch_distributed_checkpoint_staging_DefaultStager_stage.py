"""
Test purpose: Verify stage behavior.
API name: torch.distributed.checkpoint.staging.DefaultStager.stage
API signature: stage(self, state_dict, **kwargs)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| CPU snapshot | Stage an accelerator tensor, mutate the original and check that the CPU snapshot is independent. |

Execution: Single-process unittest.
Scope and prerequisites:
- The blocking path isolates snapshot independence. Future completion has separate coverage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestStage(SupplementTestCase):
    def test_stage_device_snapshot(self):
        import torch
        from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions

        device = torch.accelerator.current_accelerator()
        if device is None:
            self.skipTest("An accelerator is required for an accelerator-to-host snapshot")
        stager = DefaultStager(
            StagingOptions(
                use_async_staging=False,
                use_pinned_memory=False,
                use_shared_memory=False,
                use_non_blocking_copy=False,
            )
        )
        try:
            source = torch.tensor([3, 7], device=device)
            snapshot = stager.stage({"weight": source, "step": 2})
            source.zero_()
            self.assertEqual(snapshot["weight"].device.type, "cpu")
            self.assertEqual(snapshot["weight"].tolist(), [3, 7])
            self.assertEqual(snapshot["step"], 2)
        finally:
            stager.close()


if __name__ == "__main__":
    run_tests()
