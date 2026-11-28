"""
Test purpose: Verify _MaskPartial behavior.
API name: torch.distributed.tensor.placement_types._MaskPartial
API signature: _MaskPartial(reduce_op=None, mask_buffer=None, offset_shape=None, offset_dim=0,
    *args, **kwargs)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Masked indices and rank ownership | Mask partitioned embedding indices and reconstruct their owner ranks. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- A one-dimensional embedding is sufficient to distinguish ownership on both workers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_partition_and_reduce(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import _MaskPartial

    mesh = DeviceMesh(device.type, [0, 1])
    placement = _MaskPartial(offset_shape=torch.Size([6]))
    indices = torch.tensor([0, 2, 3, 5], device=device)
    local = placement._partition_value(indices, mesh, 0)
    case.assertEqual(local.tolist(), [0, 2, 0, 0] if rank == 0 else [0, 0, 0, 2])
    owners = placement._reduce_value(torch.full_like(local, rank + 1), mesh, 0)
    case.assertEqual(owners.tolist(), [1, 1, 2, 2])
    case.assertIsNone(placement.mask_buffer.data)


class TestMaskPartial(TwoRankTestCase):
    def test_mask_partial_partition_and_reduce(self):
        self.run_on_two_ranks(check_partition_and_reduce)


if __name__ == "__main__":
    run_tests()
