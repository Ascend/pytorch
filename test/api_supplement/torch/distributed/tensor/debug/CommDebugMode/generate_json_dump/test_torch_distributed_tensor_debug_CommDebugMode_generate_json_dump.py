"""
Test purpose: Verify generate_json_dump behavior.
API name: torch.distributed.tensor.debug.CommDebugMode.generate_json_dump
API signature: generate_json_dump(self, file_name='comm_mode_log.json', noise_level=3)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| JSON output and noise levels | Check root and child module traces at two noise levels. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Each rank uses a temporary file. The assertions cover the JSON hierarchy, not presentation
  formatting.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_trace_contents(case, rank, device):
    import json
    import tempfile

    import torch
    from torch.distributed.tensor.debug import CommDebugMode

    mode = CommDebugMode()
    module = torch.nn.Linear(4, 2, device=device)
    with mode:
        module(torch.ones(1, 4, device=device))
    with tempfile.TemporaryDirectory() as directory:
        for level in (0, 3):
            path = Path(directory) / f"trace-{level}.json"
            mode.generate_json_dump(file_name=str(path), noise_level=level)
            trace = json.loads(path.read_text())
            case.assertEqual(trace["fqn"], "Global")
            case.assertTrue(any(child["fqn"].endswith("Linear") for child in trace["children"]))


class TestGenerateJsonDump(TwoRankTestCase):
    def test_generate_json_dump_trace_contents(self):
        self.run_on_two_ranks(check_trace_contents)


if __name__ == "__main__":
    run_tests()
