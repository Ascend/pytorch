"""
Test purpose: Verify launch behavior.
API name: torch.distributed.launch.launch
API signature: launch(args)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Worker options | Check argument forwarding and reject --no-python without --use-env before starting a worker. |

Execution: Single-process unittest.
Scope and prerequisites:
- The downstream launch call is intercepted to avoid starting an unrelated training process.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestLaunch(SupplementTestCase):
    def test_launch_worker_launch_options(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from torch.distributed.launch import launch

        with patch("torch.distributed.launch.run") as run:
            for no_python, use_env in ((False, False), (False, True), (True, True)):
                args = SimpleNamespace(no_python=no_python, use_env=use_env)
                launch(args)
                run.assert_called_with(args)
            run.reset_mock()
            with self.assertRaisesRegex(ValueError, "--use-env"):
                launch(SimpleNamespace(no_python=True, use_env=False))
            run.assert_not_called()


if __name__ == "__main__":
    run_tests()
