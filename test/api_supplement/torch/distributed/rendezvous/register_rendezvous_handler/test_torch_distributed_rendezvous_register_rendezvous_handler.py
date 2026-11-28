"""
Test purpose: Verify register_rendezvous_handler behavior.
API name: torch.distributed.rendezvous.register_rendezvous_handler
API signature: register_rendezvous_handler(scheme, handler)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Handler registration and duplicates | Resolve a scheme, reject duplicates and restore state. |

Execution: Single-process unittest.
Scope and prerequisites:
- The registry is restored after the test; the handler does not establish a network rendezvous.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestRegisterRendezvousHandler(SupplementTestCase):
    def test_register_rendezvous_handler_register_and_resolve(self):
        import importlib
        from unittest.mock import patch

        module = importlib.import_module("torch.distributed.rendezvous")
        scheme = "supplement"

        def handler(url, **kwargs):
            yield url, kwargs

        with patch.dict(module._rendezvous_handlers):
            module.register_rendezvous_handler(scheme, handler)
            self.assertIs(module._rendezvous_handlers[scheme], handler)
            with self.assertRaisesRegex(RuntimeError, "already"):
                module.register_rendezvous_handler(scheme, handler)
            self.assertEqual(
                next(module._rendezvous_handlers[scheme]("supplement://job")),
                ("supplement://job", {}),
            )


if __name__ == "__main__":
    run_tests()
