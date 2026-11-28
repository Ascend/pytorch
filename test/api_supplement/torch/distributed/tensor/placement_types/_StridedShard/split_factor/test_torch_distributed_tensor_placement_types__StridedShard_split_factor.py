"""
Test purpose: Verify split_factor behavior.
API name: torch.distributed.tensor.placement_types._StridedShard.split_factor
Attribute: _StridedShard.split_factor

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Interleaved offsets | Changing split_factor changes the positions owned by each rank for the same global size. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_interleaved_offsets(case, rank, device):
    from torch.distributed.tensor.placement_types import _StridedShard
    from torch.distributed.tensor.placement_types import _StridedShardOffsetMode as Mode

    for factor, offsets in (
        (1, list(range(rank * 4, rank * 4 + 4))),
        (2, ([0, 1, 4, 5], [2, 3, 6, 7])[rank]),
    ):
        placement = _StridedShard(0, split_factor=factor)
        case.assertEqual(placement.split_factor, factor)
        case.assertEqual(placement.local_shard_size_and_offset(8, 2, rank, Mode.ALL), (4, offsets))


class TestSplitFactor(TwoRankTestCase):
    def test_split_factor_interleaved_offsets(self):
        self.run_on_two_ranks(check_interleaved_offsets)


if __name__ == "__main__":
    run_tests()
