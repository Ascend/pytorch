"""
Test purpose: Verify get_parameter_info behavior.
API name: torch.distributed.tensor.debug.CommDebugMode.get_parameter_info
API signature: get_parameter_info(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Parameter trace contents | Check that the trace lists the weight alone. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_parameters_seen_in_forward(case, rank, device):
    import torch
    from torch.distributed.tensor.debug import CommDebugMode

    mode = CommDebugMode()
    module = torch.nn.Linear(4, 2, bias=False, device=device)
    case.assertFalse(mode.get_parameter_info())
    with mode:
        module(torch.ones(1, 4, device=device))
    (params,) = mode.get_parameter_info().values()
    case.assertEqual(set(params), {"weight"})
    torch.testing.assert_close(params["weight"], module.weight)


class TestGetParameterInfo(TwoRankTestCase):
    def test_get_parameter_info_parameters_seen_in_forward(self):
        self.run_on_two_ranks(check_parameters_seen_in_forward)


if __name__ == "__main__":
    run_tests()
