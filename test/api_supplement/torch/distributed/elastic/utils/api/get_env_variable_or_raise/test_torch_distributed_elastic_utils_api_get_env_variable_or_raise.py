"""
Test purpose: Verify get_env_variable_or_raise behavior.
API name: torch.distributed.elastic.utils.api.get_env_variable_or_raise
API signature: get_env_variable_or_raise(env_name)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Present, empty and missing values | Distinguish an absent environment variable from an explicitly empty value. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestGetEnvVariableOrRaise(SupplementTestCase):
    def test_get_env_variable_or_raise_present_and_missing(self):
        import os
        from unittest.mock import patch

        from torch.distributed.elastic.utils.api import get_env_variable_or_raise

        key = "TORCH_SUPPLEMENT_ENV_TEST"
        with patch.dict(os.environ):
            os.environ.pop(key, None)
            with self.assertRaisesRegex(ValueError, key):
                get_env_variable_or_raise(key)
            for value in ("", "0", "worker"):
                os.environ[key] = value
                self.assertEqual(get_env_variable_or_raise(env_name=key), value)


if __name__ == "__main__":
    run_tests()
