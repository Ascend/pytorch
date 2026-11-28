"""
Test purpose: Verify PrepareModuleInputOutput behavior.
API name: torch.distributed.tensor.parallel.PrepareModuleInputOutput
API signature: PrepareModuleInputOutput(*, input_layouts=None, desired_input_layouts=None,
    input_kwarg_layouts=None, desired_input_kwarg_layouts=None, use_local_input=False,
    output_layouts, desired_output_layouts, use_local_output=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Input and output placement | Gather rows and shard columns in both output modes. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The rank-distinct fixture exercises positional inputs and both output modes. Keyword layout maps
  are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_redistribute_input_and_output(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor.parallel import (
        PrepareModuleInputOutput,
        parallelize_module,
    )

    mesh = DeviceMesh(device.type, [0, 1])
    full = torch.arange(24, device=device).reshape(4, 6)
    for local_output in (False, True):
        style = PrepareModuleInputOutput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
            use_local_output=local_output,
        )
        module = parallelize_module(torch.nn.Identity(), mesh, style)
        result = module(full.chunk(2, dim=0)[rank].contiguous())
        case.assertEqual(isinstance(result, DTensor), not local_output)
        if not local_output:
            case.assertEqual(result.placements, (Shard(1),))
            result = result.to_local()
        torch.testing.assert_close(result, full.chunk(2, dim=1)[rank])


class TestPrepareModuleInputOutput(TwoRankTestCase):
    def test_prepare_module_input_output_redistribute_input_and_output(self):
        self.run_on_two_ranks(check_redistribute_input_and_output)


if __name__ == "__main__":
    run_tests()
