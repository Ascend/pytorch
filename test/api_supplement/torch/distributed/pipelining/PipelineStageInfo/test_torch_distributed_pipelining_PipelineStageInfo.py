"""
Test purpose: Verify PipelineStageInfo behavior.
API name: torch.distributed.pipelining.PipelineStageInfo
API signature: PipelineStageInfo(*, stage_index, microbatch_index, is_metadata_inference=False)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Constructor and immutability | Check metadata-inference identity, keyword-only construction and immutability. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestPipelineStageInfo(SupplementTestCase):
    def test_pipeline_stage_info_immutable_identity(self):
        from dataclasses import FrozenInstanceError

        from torch.distributed.pipelining import PipelineStageInfo

        actual = PipelineStageInfo(stage_index=2, microbatch_index=3)
        self.assertFalse(actual.is_metadata_inference)
        probe = PipelineStageInfo(stage_index=2, microbatch_index=3, is_metadata_inference=True)
        self.assertNotEqual(actual, probe)
        with self.assertRaises(FrozenInstanceError):
            actual.microbatch_index = 0
        with self.assertRaises(TypeError):
            PipelineStageInfo(2, 3)


if __name__ == "__main__":
    run_tests()
