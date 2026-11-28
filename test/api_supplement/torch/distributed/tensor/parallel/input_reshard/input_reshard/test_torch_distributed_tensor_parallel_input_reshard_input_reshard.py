"""
Test purpose: Verify input_reshard behavior.
API name: torch.distributed.tensor.parallel.input_reshard.input_reshard
API signature: input_reshard(module, tp_device_mesh, input_reshard_dim=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Saved-input redistribution | Check disabled resharding and saved-input gradients. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The square operation saves its input for backward; both tensor axes and disabled sharding are
  exercised.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_backward_restores_saved_input(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.parallel.input_reshard import input_reshard

    class Square(torch.nn.Module):
        def forward(self, value):
            return value * value

    mesh = DeviceMesh(device.type, [0, 1])
    for axis in (None, 0, 1):
        model = Square()
        case.assertIs(input_reshard(model, mesh, axis), model)
        local = torch.arange(8.0, device=device).reshape(2, 4).requires_grad_()
        value = DTensor.from_local(local, mesh, [Replicate()])
        model(value).sum().backward()
        torch.testing.assert_close(local.grad, local.detach() * 2)


class TestInputReshard(TwoRankTestCase):
    def test_input_reshard_backward_restores_saved_input(self):
        self.run_on_two_ranks(check_backward_restores_saved_input)


if __name__ == "__main__":
    run_tests()
