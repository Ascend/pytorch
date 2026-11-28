"""
Test purpose: Verify is_torchelastic_launched behavior.
API name: torch.distributed.distributed_c10d.is_torchelastic_launched
API signature: is_torchelastic_launched()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Environment variable states | Check missing, empty and populated elastic run IDs. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestIsTorchelasticLaunched(SupplementTestCase):
    def test_is_torchelastic_launched_launcher_environment(self):
        import os
        from unittest.mock import patch

        from torch.distributed.distributed_c10d import is_torchelastic_launched

        with patch.dict(os.environ):
            os.environ.pop("TORCHELASTIC_RUN_ID", None)
            self.assertFalse(is_torchelastic_launched())
            for value in ("", "local-job"):
                os.environ["TORCHELASTIC_RUN_ID"] = value
                self.assertTrue(is_torchelastic_launched())


if __name__ == "__main__":
    run_tests()
