"""
Test purpose: Verify parse_min_max_nnodes behavior.
API name: torch.distributed.run.parse_min_max_nnodes
API signature: parse_min_max_nnodes(nnodes)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Fixed and ranged node counts | Parse fixed and ranged counts and check syntax and value errors. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestParseMinMaxNnodes(SupplementTestCase):
    def test_parse_min_max_nnodes_fixed_and_elastic_sizes(self):
        from torch.distributed.run import parse_min_max_nnodes

        for text, expected in (("1", (1, 1)), ("2:5", (2, 5)), ("3:3", (3, 3))):
            self.assertEqual(parse_min_max_nnodes(text), expected)
        with self.assertRaises(ValueError):
            parse_min_max_nnodes("1:")
        with self.assertRaises(RuntimeError):
            parse_min_max_nnodes("1:2:3")


if __name__ == "__main__":
    run_tests()
