"""
Test purpose: Verify slot_for behavior.
API name: torch.distributed.pipelining.schedules.PipelineActivationLiveness.slot_for
API signature: slot_for(self, stage_index, microbatch_index)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Slot reuse and missing microbatches | Look up a reused slot across microbatches and reject an unknown microbatch. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestSlotFor(SupplementTestCase):
    def test_slot_for_reused_slot(self):
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
        self.assertEqual(plan.slot_for(0, 0), 0)
        self.assertEqual(plan.slot_for(0, 1), 0)
        with self.assertRaisesRegex(ValueError, "slot"):
            plan.slot_for(0, 2)


if __name__ == "__main__":
    run_tests()
