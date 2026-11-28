"""
Test purpose: Verify ALL_REDUCE_OPS behavior.
API name: torch.distributed.tensor.placement_types.Partial.ALL_REDUCE_OPS
Attribute: Partial.ALL_REDUCE_OPS

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Reduction names and materialization | Check reduction names and materialize the rank-local sum. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_supported_operations(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Partial, Replicate

    case.assertEqual(set(Partial.ALL_REDUCE_OPS), {"sum", "avg", "min", "max", "product"})
    mesh = DeviceMesh(device.type, [0, 1])
    partial = DTensor.from_local(torch.tensor([rank + 1.0], device=device), mesh, [Partial("sum")])
    case.assertEqual(partial.redistribute(placements=[Replicate()]).to_local().item(), 3)


class TestALLREDUCEOPS(TwoRankTestCase):
    def test_all_reduce_ops_supported_operations(self):
        self.run_on_two_ranks(check_supported_operations)


if __name__ == "__main__":
    run_tests()
