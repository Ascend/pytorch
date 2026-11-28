"""
Test purpose: Verify mesh_dim_names behavior.
API name: torch.distributed.device_mesh.DeviceMesh.mesh_dim_names
Attribute: DeviceMesh.mesh_dim_names

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Named and unnamed dimensions | Check default names, multidimensional names and projection onto a named dimension. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The two-rank mesh checks public topology metadata; no tensor payload or numerical tolerance is
  involved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_named_and_unnamed(case, rank, device):
    from torch.distributed.device_mesh import DeviceMesh

    case.assertIsNone(DeviceMesh(device.type, [0, 1]).mesh_dim_names)
    mesh = DeviceMesh(device.type, [[0, 1]], mesh_dim_names=("replica", "shard"))
    case.assertEqual(mesh.mesh_dim_names, ("replica", "shard"))
    case.assertEqual(mesh["shard"].mesh_dim_names, ("shard",))


class TestMeshDimNames(TwoRankTestCase):
    def test_mesh_dim_names_named_and_unnamed(self):
        self.run_on_two_ranks(check_named_and_unnamed)


if __name__ == "__main__":
    run_tests()
