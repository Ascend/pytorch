"""
Test purpose: Verify register_backend behavior.
API name: torch.distributed.rpc.backend_registry.register_backend
API signature: register_backend(backend_name, construct_rpc_backend_options_handler,
    init_backend_handler)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Backend factory and duplicate registration | Call the registered options factory and reject duplicates. |

Execution: Single-process unittest.
Scope and prerequisites:
- The backend registry is isolated; registering a backend does not start an RPC agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestRegisterBackend(SupplementTestCase):
    def test_register_backend_handler_registration(self):
        from unittest.mock import Mock, patch

        from torch.distributed.rpc import backend_registry as registry

        factory = Mock(return_value=object())
        init = Mock()
        with patch.object(registry, "BackendType", registry.BackendType):
            backend = registry.register_backend("SUPPLEMENT", factory, init)
            self.assertTrue(registry.backend_registered("SUPPLEMENT"))
            self.assertIs(registry.construct_rpc_backend_options(backend), factory.return_value)
            factory.assert_called_once()
            init.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                registry.register_backend("SUPPLEMENT", factory, init)


if __name__ == "__main__":
    run_tests()
