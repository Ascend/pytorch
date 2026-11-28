"""
Test purpose: Verify run_script_path behavior.
API name: torch.distributed.run.run_script_path
API signature: run_script_path(training_script, *training_script_args)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Script arguments | Execute a temporary training script as __main__ and check its exact argument vector. |

Execution: Single-process unittest.
Scope and prerequisites:
- Execution is confined to a temporary script and its argument vector is restored afterwards.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestRunScriptPath(SupplementTestCase):
    def test_run_script_path_main_arguments(self):
        import json
        import tempfile
        from unittest.mock import patch

        from torch.distributed.run import run_script_path

        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "argv", []):
            script = Path(directory) / "train.py"
            output = Path(directory) / "result.json"
            script.write_text(
                "import json, sys\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps([__name__, sys.argv]))\n"
            )
            run_script_path(str(script), str(output), "two words")
            self.assertEqual(
                json.loads(output.read_text()),
                ["__main__", [str(script), str(output), "two words"]],
            )


if __name__ == "__main__":
    run_tests()
