"""
Test purpose: Verify backend_registered behavior.
API name: torch.distributed.rpc.backend_registry.backend_registered
API signature: backend_registered(backend_name)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Registered and missing backend names | Look up known names from the runtime registry and a missing name. |

Execution: Single-process unittest.
Scope and prerequisites:
- Registry lookup does not start an RPC agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestBackendRegistered(SupplementTestCase):
    def test_backend_registered_case_sensitive_lookup(self):
        from torch.distributed.rpc import backend_registry as registry

        for backend in registry.BackendType:
            self.assertTrue(registry.backend_registered(backend.name))
        self.assertFalse(registry.backend_registered("MISSING_SUPPLEMENT_BACKEND"))
        self.assertFalse(registry.backend_registered(""))


if __name__ == "__main__":
    run_tests()
