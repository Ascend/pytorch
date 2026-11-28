"""
Test purpose: Verify unbox behavior.
API name: torch.distributed.ReduceOp.unbox
API signature: unbox(obj: ScriptObject) -> ReduceOp

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Valid and invalid boxed input | Round-trip several native reduction operators through a boxed TorchScript value. |

Execution: Single-process unittest.
Scope and prerequisites:
- Invalid boxed input is checked in addition to the reduction-kind round trip.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestUnbox(SupplementTestCase):
    def test_unbox_reduction_round_trip(self):
        from torch.distributed import ReduceOp

        for kind in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PRODUCT):
            boxed = ReduceOp(kind).boxed()
            restored = ReduceOp.unbox(boxed)
            self.assertEqual(restored.op, kind)
        with self.assertRaises((TypeError, RuntimeError)):
            ReduceOp.unbox(None)


if __name__ == "__main__":
    run_tests()
