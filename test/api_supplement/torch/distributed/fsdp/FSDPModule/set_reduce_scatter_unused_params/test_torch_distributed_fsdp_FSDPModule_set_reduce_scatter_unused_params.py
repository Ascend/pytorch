"""
Test purpose: Verify set_reduce_scatter_unused_params behavior.
API name: torch.distributed.fsdp.FSDPModule.set_reduce_scatter_unused_params
API signature: set_reduce_scatter_unused_params(self, reduce_scatter_unused_params, *,
    recurse=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Unused parameter reduction | Check zero-gradient participation when only one rank uses a parameter. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_unused_parameter(case, rank, device):
    import torch
    from torch import nn
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    class Conditional(nn.Module):
        def __init__(self):
            super().__init__()
            self.always = nn.Parameter(torch.ones(4, device=device))
            self.optional = nn.Parameter(torch.ones(4, device=device))

        def forward(self, use_optional):
            result = self.always.sum()
            return result + self.optional.sum() if use_optional else result

    model = Conditional()
    fully_shard(model, mesh=DeviceMesh(device.type, [0, 1]))
    model.set_reduce_scatter_unused_params(True)
    model(rank == 0).backward()
    torch.testing.assert_close(model.always.grad.to_local(), torch.ones(2, device=device))
    torch.testing.assert_close(model.optional.grad.to_local(), torch.full((2,), 0.5, device=device))


class TestSetReduceScatterUnusedParams(TwoRankTestCase):
    def test_set_reduce_scatter_unused_params_unused_parameter(self):
        self.run_on_two_ranks(check_unused_parameter)


if __name__ == "__main__":
    run_tests()
