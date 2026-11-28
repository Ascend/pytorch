"""
Test purpose: Verify abort behavior.
API name: torch.distributed.device_mesh.DeviceMesh.abort
API signature: abort(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Submesh abort restrictions | A submesh must reject ambiguous aborts; its parent process group remains usable. |
| Root group teardown | Abort an initialized subgroup and observe its removal from the group registry. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- An isolated subgroup owns the root mesh used for abort. The default group stays available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_submesh_scope(case, rank, device):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    mesh = DeviceMesh(device.type, [[0, 1]], mesh_dim_names=("replica", "shard"))
    with case.assertRaisesRegex(RuntimeError, "submesh"):
        mesh["shard"].abort()
    value = torch.tensor([rank + 1], device=device)
    dist.all_reduce(value, group=mesh["shard"].get_group())
    case.assertEqual(value.item(), 3)


def check_root_group(case, rank, device):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    group = dist.new_group([0, 1])
    mesh = DeviceMesh.from_group(group, device.type)
    dist.barrier(group=group)
    mesh.abort()
    with case.assertRaises(KeyError):
        dist.get_process_group_ranks(group)
    value = torch.tensor([rank + 1], device=device)
    dist.all_reduce(value)
    case.assertEqual(value.item(), 3)


class TestAbort(TwoRankTestCase):
    def test_abort_root_group(self):
        self.run_on_two_ranks(check_root_group)

    def test_abort_submesh_scope(self):
        self.run_on_two_ranks(check_submesh_scope)


if __name__ == "__main__":
    run_tests()
