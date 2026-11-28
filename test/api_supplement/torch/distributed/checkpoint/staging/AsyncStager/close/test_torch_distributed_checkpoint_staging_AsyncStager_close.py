"""
Test purpose: Verify close behavior.
API name: torch.distributed.checkpoint.staging.AsyncStager.close
API signature: close(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Inherited cleanup | A minimal concrete stager inherits the default close and preserves its application state. |

Execution: Single-process unittest.
Scope and prerequisites:
- The base protocol has no executor. Concrete staging and shutdown have separate tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestClose(SupplementTestCase):
    def test_close_inherited_cleanup(self):
        from torch.distributed.checkpoint.staging import AsyncStager

        class MemoryStager(AsyncStager):
            def stage(self, state_dict):
                return state_dict

        stager = MemoryStager()
        state = {"step": 2}
        self.assertIs(stager.stage(state), state)
        self.assertIsNone(stager.close())
        self.assertIsNone(stager.close())


if __name__ == "__main__":
    run_tests()
