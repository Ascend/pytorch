"""
Test purpose: Verify get_activation_lifetime behavior.
API name: torch.distributed.pipelining.schedules.PipelineActivationLiveness.get_activation_lifetime
API signature: get_activation_lifetime(self, stage_index, microbatch_index)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Inclusive bounds and missing stages | Check inclusive microbatch bounds and reject a missing stage. |

Execution: Single-process unittest.
Scope and prerequisites:
- This is a host-side value object. Device, gradient and mixed-device behavior are inapplicable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestGetActivationLifetime(SupplementTestCase):
    def test_get_activation_lifetime_inclusive_intervals(self):
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
        self.assertEqual(plan.get_activation_lifetime(0, 0), (0, 1))
        self.assertEqual(plan.get_activation_lifetime(0, 1), (2, 3))
        with self.assertRaisesRegex(ValueError, "lifetime"):
            plan.get_activation_lifetime(1, 0)


if __name__ == "__main__":
    run_tests()
