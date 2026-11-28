"""
Test purpose: Verify DataParallelMeshDims behavior.
API name: torch.distributed.fsdp.DataParallelMeshDims
API signature: DataParallelMeshDims(shard=None, replicate=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Dimension names and missing mesh | Normalize single and multiple dimension names and reject an unspecified mesh. |

Execution: Single-process unittest.
Scope and prerequisites:
- This host-side contract has no device or gradient inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import SupplementTestCase, run_tests

sys.path.pop(0)


class TestDataParallelMeshDims(SupplementTestCase):
    def test_data_parallel_mesh_dims_named_dimensions(self):
        from torch.distributed.fsdp import DataParallelMeshDims

        for shard, replica, expected in (
            ("x", None, (("x",), ())),
            (("x", "y"), "z", (("x", "y"), ("z",))),
        ):
            dims = DataParallelMeshDims(shard=shard, replicate=replica)
            self.assertEqual((dims.shard_names, dims.replicate_names), expected)
        with self.assertRaisesRegex(ValueError, "shard or replicate"):
            DataParallelMeshDims()


if __name__ == "__main__":
    run_tests()
