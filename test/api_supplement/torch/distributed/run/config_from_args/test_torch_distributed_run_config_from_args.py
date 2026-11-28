"""
Test purpose: Verify config_from_args behavior.
API name: torch.distributed.run.config_from_args
API signature: config_from_args(args)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Elastic launch configuration | Check node bounds, per-node worker count, script interpreter and argument order. |

Execution: Single-process unittest.
Scope and prerequisites:
- Configuration creation is isolated from child process launch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestConfigFromArgs(SupplementTestCase):
    def test_config_from_args_elastic_launch_config(self):
        import os
        from unittest.mock import patch

        from torch.distributed.run import config_from_args, parse_args

        with patch.dict(os.environ):
            args = parse_args(["--nnodes=1:3", "--nproc-per-node=2", "--no-python", "worker", "x"])
            config, command, argv = config_from_args(args)
        self.assertEqual((config.min_nodes, config.max_nodes, config.nproc_per_node), (1, 3, 2))
        self.assertEqual((command, argv), ("worker", ["x"]))


if __name__ == "__main__":
    run_tests()
