"""
Test purpose: Verify get_log_level behavior.
API name: torch.distributed.elastic.utils.log_level.get_log_level
API signature: get_log_level()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Default log severity | Check that the default is a valid Python logging level at WARNING severity. |

Execution: Single-process unittest.
Scope and prerequisites:
- The function takes no arguments or device input.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestGetLogLevel(SupplementTestCase):
    def test_get_log_level_default_severity(self):
        import logging

        from torch.distributed.elastic.utils.log_level import get_log_level

        self.assertEqual(get_log_level(), "WARNING")
        self.assertEqual(logging.getLevelName(get_log_level()), logging.WARNING)


if __name__ == "__main__":
    run_tests()
