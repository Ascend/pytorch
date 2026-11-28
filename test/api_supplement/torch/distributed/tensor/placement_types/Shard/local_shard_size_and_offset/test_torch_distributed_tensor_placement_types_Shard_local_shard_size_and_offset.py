"""
Test purpose: Verify local_shard_size_and_offset behavior.
API name: torch.distributed.tensor.placement_types.Shard.local_shard_size_and_offset
API signature: local_shard_size_and_offset(curr_local_size, num_chunks, rank)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Empty and uneven partitions | Check shard bounds for even, uneven, singleton and empty inputs. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses a two-rank mesh. Cross-host topologies and performance are outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_chunk_boundaries(case, rank, device):
    from torch.distributed.tensor import Shard

    for total in (0, 1, 4, 5):
        width = (total + 1) // 2
        offset = min(rank * width, total)
        size = min(width, total - offset)
        case.assertEqual(Shard.local_shard_size_and_offset(total, 2, rank), (size, offset))


class TestLocalShardSizeAndOffset(TwoRankTestCase):
    def test_local_shard_size_and_offset_chunk_boundaries(self):
        self.run_on_two_ranks(check_chunk_boundaries)


if __name__ == "__main__":
    run_tests()
