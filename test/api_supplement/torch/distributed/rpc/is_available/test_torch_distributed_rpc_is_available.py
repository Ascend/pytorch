"""
Test purpose: Verify is_available behavior.
API name: torch.distributed.rpc.is_available
API signature: is_available()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Build capability | Compare RPC availability with the native RPC initialization entry point. |

Execution: Single-process unittest.
Scope and prerequisites:
- The build capability is independent of whether an RPC agent is running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestIsAvailable(SupplementTestCase):
    def test_is_available_build_capability(self):
        import torch
        from torch.distributed import rpc

        self.assertIs(rpc.is_available(), hasattr(torch._C, "_rpc_init"))
        with self.assertRaises(TypeError):
            rpc.is_available("rpc")


if __name__ == "__main__":
    run_tests()
