"""
Test purpose: Verify offset_dim behavior.
API name: torch.distributed.tensor.placement_types._MaskPartial.offset_dim
Attribute: _MaskPartial.offset_dim

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Embedding axis selection | Select different embedding axes and check ownership masks against their shard bounds. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_axis_selection(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import _MaskPartial

    mesh = DeviceMesh(device.type, [0, 1])
    for axis, size in ((0, 4), (1, 8), (-1, 8)):
        placement = _MaskPartial(offset_dim=axis, offset_shape=torch.Size([4, 8]))
        case.assertEqual(placement.offset_dim, axis)
        indices = torch.arange(size, device=device)
        placement._partition_value(indices, mesh, 0)
        expected = [
            not (rank * (size // 2) <= index < (rank + 1) * (size // 2)) for index in range(size)
        ]
        case.assertEqual(placement.mask_buffer.data.tolist(), expected)


class TestOffsetDim(TwoRankTestCase):
    def test_offset_dim_axis_selection(self):
        self.run_on_two_ranks(check_axis_selection)


if __name__ == "__main__":
    run_tests()
