"""
Test purpose: Verify implicit_replication behavior.
API name: torch.distributed.tensor.experimental.implicit_replication
API signature: implicit_replication()

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Context scope and restoration | Mixed Tensor/DTensor operations accept replicated input only within the context. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_scope_restoration(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.experimental import implicit_replication

    mesh = DeviceMesh(device.type, [0, 1])
    value = torch.tensor([2.0, 3.0], device=device)
    distributed = DTensor.from_local(value, mesh, [Replicate()])
    with implicit_replication():
        case.assertEqual((distributed + value).to_local().tolist(), [4.0, 6.0])
    with case.assertRaisesRegex(RuntimeError, "Tensor"):
        distributed + value


class TestImplicitReplication(TwoRankTestCase):
    def test_implicit_replication_scope_restoration(self):
        self.run_on_two_ranks(check_scope_restoration)


if __name__ == "__main__":
    run_tests()
