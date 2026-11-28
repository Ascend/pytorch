"""
Test purpose: Verify set_reshard_after_forward behavior.
API name: torch.distributed.fsdp.FSDPModule.set_reshard_after_forward
API signature: set_reshard_after_forward(self, reshard_after_forward, recurse=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Resharding policy | Check parameter layout after forward under both policies. |

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


def check_forward_parameter_layout(case, rank, device):
    import torch
    from distributed_test_support import (
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    from torch.distributed.tensor import DTensor

    for reshard in (True, False):
        model.zero_grad(set_to_none=True)
        model.set_reshard_after_forward(reshard)
        result = model(torch.full((1, 4), float(rank + 1), device=device))
        case.assertEqual(isinstance(model.weight, DTensor), reshard)
        result.sum().backward()
        check_reduced_gradient(model)


class TestSetReshardAfterForward(TwoRankTestCase):
    def test_set_reshard_after_forward_forward_parameter_layout(self):
        self.run_on_two_ranks(check_forward_parameter_layout)


if __name__ == "__main__":
    run_tests()
