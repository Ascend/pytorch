"""
Test purpose: Verify PipelineActivationLiveness behavior.
API name: torch.distributed.pipelining.schedules.PipelineActivationLiveness
API signature: PipelineActivationLiveness(pp_rank, granularity, stage_indices, num_microbatches,
    slot_by_stage_and_microbatch, num_slots, _activation_lifetime_by_stage_and_microbatch)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Plan metadata and immutability | Check the plan's stage set, slot count and immutable dataclass fields. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestPipelineActivationLiveness(SupplementTestCase):
    def test_pipeline_activation_liveness_immutable_plan(self):
        from torch.distributed.pipelining.schedules import PipelineActivationLiveness

        plan = PipelineActivationLiveness(
            pp_rank=0,
            granularity="stage_microbatch",
            stage_indices=(0,),
            num_microbatches=2,
            slot_by_stage_and_microbatch={(0, 0): 0, (0, 1): 0},
            num_slots=1,
            _activation_lifetime_by_stage_and_microbatch={(0, 0): (0, 1), (0, 1): (2, 3)},
        )
        from dataclasses import FrozenInstanceError

        self.assertEqual(plan.stage_indices, (0,))
        self.assertEqual(plan.num_slots, 1)
        self.assertEqual(plan.num_microbatches, 2)
        with self.assertRaises(FrozenInstanceError):
            plan.num_slots = 3


if __name__ == "__main__":
    run_tests()
