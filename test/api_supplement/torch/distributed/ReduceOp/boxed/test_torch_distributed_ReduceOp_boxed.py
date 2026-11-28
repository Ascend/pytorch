"""
Test purpose: Verify boxed behavior.
API name: torch.distributed.ReduceOp.boxed
API signature: boxed(self) -> ScriptObject

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Boxed value round trip | Round-trip several native reduction operators through a boxed TorchScript value. |

Execution: Single-process unittest.
Scope and prerequisites:
- The boxed values are inspected via unbox; neither conversion launches a collective.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestBoxed(SupplementTestCase):
    def test_boxed_reduction_round_trip(self):
        from torch.distributed import ReduceOp

        for kind in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN, ReduceOp.PRODUCT):
            boxed = ReduceOp(kind).boxed()
            restored = ReduceOp.unbox(boxed)
            self.assertEqual(restored.op, kind)
        self.assertNotEqual(ReduceOp.unbox(ReduceOp(ReduceOp.MAX).boxed()).op, ReduceOp.SUM)


if __name__ == "__main__":
    run_tests()
