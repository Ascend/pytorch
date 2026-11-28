"""
Test purpose: Verify get_sharding_info behavior.
API name: torch.distributed.tensor.debug.CommDebugMode.get_sharding_info
API signature: get_sharding_info(self)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Placement metadata | Compare traced and actual column-sharded weight placements. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_sharded_parameter_metadata(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.debug import CommDebugMode
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module

    mesh = DeviceMesh(device.type, [0, 1])
    module = parallelize_module(
        torch.nn.Linear(4, 4, bias=False, device=device), mesh, ColwiseParallel()
    )
    mode = CommDebugMode()
    case.assertFalse(mode.get_sharding_info())
    with mode:
        module(torch.ones(1, 4, device=device))
    ((name, layout),) = mode.get_sharding_info().items()
    case.assertTrue(name.endswith(".weight"))
    case.assertEqual(tuple(layout), (Shard(0),))
    case.assertEqual(tuple(layout), module.weight.placements)


class TestGetShardingInfo(TwoRankTestCase):
    def test_get_sharding_info_sharded_parameter_metadata(self):
        self.run_on_two_ranks(check_sharded_parameter_metadata)


if __name__ == "__main__":
    run_tests()
