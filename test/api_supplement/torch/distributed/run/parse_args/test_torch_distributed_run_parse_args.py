"""
Test purpose: Verify parse_args behavior.
API name: torch.distributed.run.parse_args
API signature: parse_args(args)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Script argument boundary | Distinguish launcher flags and a similarly named option passed after the script. |

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
    def test_parse_args_script_remainder(self):
        from torch.distributed.run import parse_args

        args = parse_args(["--nnodes=2", "--nproc-per-node=3", "train.py", "--nnodes", "9"])
        self.assertEqual(
            (args.nnodes, args.nproc_per_node, args.training_script), ("2", "3", "train.py")
        )
        self.assertEqual(args.training_script_args, ["--nnodes", "9"])


if __name__ == "__main__":
    run_tests()
