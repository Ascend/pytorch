"""
Test purpose: Verify offset_shape behavior.
API name: torch.distributed.tensor.placement_types._MaskPartial.offset_shape
Attribute: _MaskPartial.offset_shape

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Missing shape and uneven shards | Require the global shape for partitioning and respect the final short shard. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_missing_and_uneven_shape(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import _MaskPartial

    mesh = DeviceMesh(device.type, [0, 1])
    indices = torch.arange(5, device=device)
    case.assertIsNone(_MaskPartial().offset_shape)
    with case.assertRaisesRegex(AssertionError, "offset_shape"):
        _MaskPartial()._partition_value(indices, mesh, 0)
    placement = _MaskPartial(offset_shape=torch.Size([5]))
    placement._partition_value(indices, mesh, 0)
    case.assertEqual(
        placement.mask_buffer.data.tolist(),
        [False, False, False, True, True] if rank == 0 else [True, True, True, False, False],
    )


class TestOffsetShape(TwoRankTestCase):
    def test_offset_shape_missing_and_uneven_shape(self):
        self.run_on_two_ranks(check_missing_and_uneven_shape)


if __name__ == "__main__":
    run_tests()
