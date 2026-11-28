"""
Test purpose: Verify mask_buffer behavior.
API name: torch.distributed.tensor.placement_types._MaskPartial.mask_buffer
Attribute: _MaskPartial.mask_buffer

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Mask buffer lifetime | The default mask belongs to one placement and is released after reduction. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The buffer is observed on concrete placement instances, not through a class attribute.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_buffer_lifetime(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import _MaskPartial

    mesh = DeviceMesh(device.type, [0, 1])
    placement = _MaskPartial(offset_shape=torch.Size([4]))
    unused = _MaskPartial()
    case.assertIsNot(placement.mask_buffer, unused.mask_buffer)
    indices = torch.arange(4, device=device)
    placement._partition_value(indices, mesh, 0)
    case.assertEqual(
        placement.mask_buffer.data.tolist(),
        [False, False, True, True] if rank == 0 else [True, True, False, False],
    )
    case.assertIsNone(unused.mask_buffer.data)
    placement._reduce_value(torch.ones_like(indices), mesh, 0).wait()
    case.assertIsNone(placement.mask_buffer.data)


class TestMaskBuffer(TwoRankTestCase):
    def test_mask_buffer_buffer_lifetime(self):
        self.run_on_two_ranks(check_buffer_lifetime)


if __name__ == "__main__":
    run_tests()
