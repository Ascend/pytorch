"""
Test purpose: Verify local_shard_size_and_offset behavior.
API name: torch.distributed.tensor.placement_types._StridedShard.local_shard_size_and_offset
API signature: local_shard_size_and_offset(self, curr_local_size, num_chunks, rank,
    offset_mode=<_StridedShardOffsetMode.FIRST: 0>)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Full and first-only offsets | Check full and first-only offsets, including empty input. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_offset_modes(case, rank, device):
    from torch.distributed.tensor.placement_types import _StridedShard
    from torch.distributed.tensor.placement_types import _StridedShardOffsetMode as Mode

    placement = _StridedShard(0, split_factor=2)
    expected = ([0, 1, 4, 5], [2, 3, 6, 7])[rank]
    case.assertEqual(placement.local_shard_size_and_offset(8, 2, rank, Mode.ALL), (4, expected))
    case.assertEqual(placement.local_shard_size_and_offset(8, 2, rank), (4, expected[0]))
    case.assertEqual(placement.local_shard_size_and_offset(0, 2, rank, Mode.NONE), (0, None))


class TestLocalShardSizeAndOffset(TwoRankTestCase):
    def test_local_shard_size_and_offset_offset_modes(self):
        self.run_on_two_ranks(check_offset_modes)


if __name__ == "__main__":
    run_tests()
