"""
Test purpose: Verify placements behavior.
API name: torch.distributed.tensor.DTensor.placements
Attribute: DTensor.placements

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Redistribution and source layout | Replicate a shard and check that the source layout is preserved. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank accelerator process group on one host.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_redistributed_layout(case, rank, device):
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    mesh = DeviceMesh(device.type, [0, 1])
    source = DTensor.from_local(torch.tensor([rank], device=device), mesh, [Shard(0)])
    replica = source.redistribute(placements=[Replicate()])
    case.assertEqual(source.placements, (Shard(0),))
    case.assertEqual(replica.placements, (Replicate(),))
    case.assertEqual(replica.to_local().tolist(), [0, 1])


class TestPlacements(TwoRankTestCase):
    def test_placements_redistributed_layout(self):
        self.run_on_two_ranks(check_redistributed_layout)


if __name__ == "__main__":
    run_tests()
