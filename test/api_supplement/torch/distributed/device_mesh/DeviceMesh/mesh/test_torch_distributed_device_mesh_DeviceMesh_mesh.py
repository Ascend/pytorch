"""
Test purpose: Verify mesh behavior.
API name: torch.distributed.device_mesh.DeviceMesh.mesh
Attribute: DeviceMesh.mesh

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Mesh shape and rank order | Preserve rank order, mesh shape and coordinates on both workers. |

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


def check_rank_layout(case, rank, device):
    from torch.distributed.device_mesh import DeviceMesh

    mesh = DeviceMesh(device.type, [[0, 1]], mesh_dim_names=("replica", "shard"))
    case.assertEqual(mesh.mesh.tolist(), [[0, 1]])
    case.assertEqual(tuple(mesh.get_coordinate()), (0, rank))
    case.assertEqual(mesh["shard"].mesh.tolist(), [0, 1])


class TestMesh(TwoRankTestCase):
    def test_mesh_rank_layout(self):
        self.run_on_two_ranks(check_rank_layout)


if __name__ == "__main__":
    run_tests()
