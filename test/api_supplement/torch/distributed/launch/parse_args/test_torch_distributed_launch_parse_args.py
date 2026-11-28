"""
Test purpose: Verify parse_args behavior.
API name: torch.distributed.launch.parse_args
API signature: parse_args(args)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Flag aliases and defaults | Check legacy flag spelling, default launch mode and preservation of script arguments. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestParseArgs(SupplementTestCase):
    def test_parse_args_use_env_aliases(self):
        from torch.distributed.launch import parse_args

        for flags, expected in (([], False), (["--use-env"], True), (["--use_env"], True)):
            args = parse_args([*flags, "train.py", "--epochs", "2"])
            self.assertIs(args.use_env, expected)
            self.assertEqual(args.training_script, "train.py")
            self.assertEqual(args.training_script_args, ["--epochs", "2"])


if __name__ == "__main__":
    run_tests()
