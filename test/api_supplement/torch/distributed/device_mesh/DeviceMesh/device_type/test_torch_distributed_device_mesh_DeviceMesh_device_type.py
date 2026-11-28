"""
Test purpose: Verify device_type behavior.
API name: torch.distributed.device_mesh.DeviceMesh.device_type
Attribute: DeviceMesh.device_type

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Device type and submeshes | Check device type on a root mesh and a named submesh. |

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


def check_mesh_device(case, rank, device):
    from torch.distributed.device_mesh import DeviceMesh

    mesh = DeviceMesh(device.type, [[0, 1]], mesh_dim_names=("replica", "shard"))
    case.assertEqual(mesh.device_type, device.type)
    case.assertEqual(mesh["shard"].device_type, device.type)


class TestDeviceType(TwoRankTestCase):
    def test_device_type_mesh_device(self):
        self.run_on_two_ranks(check_mesh_device)


if __name__ == "__main__":
    run_tests()
