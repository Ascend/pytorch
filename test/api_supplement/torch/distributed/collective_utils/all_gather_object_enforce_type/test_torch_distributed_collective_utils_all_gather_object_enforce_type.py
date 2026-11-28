"""
Test purpose: Verify all_gather_object_enforce_type behavior.
API name: torch.distributed.collective_utils.all_gather_object_enforce_type
API signature: all_gather_object_enforce_type(pg, object_list, obj, type_checker=<same-type
    predicate>)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Object order and type checking | Check rank order and reject mixed types on both ranks. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The default checker is exercised with homogeneous and heterogeneous objects.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_rank_order_and_types(case, rank, device):
    import torch.distributed as dist
    from torch.distributed.collective_utils import all_gather_object_enforce_type

    values = [None, None]
    all_gather_object_enforce_type(dist.group.WORLD, values, {"rank": rank})
    case.assertEqual(values, [{"rank": 0}, {"rank": 1}])
    with case.assertRaisesRegex(TypeError, "index 1"):
        all_gather_object_enforce_type(dist.group.WORLD, values, rank if rank == 0 else "1")


class TestAllGatherObjectEnforceType(TwoRankTestCase):
    def test_all_gather_object_enforce_type_rank_order_and_types(self):
        self.run_on_two_ranks(check_rank_order_and_types)


if __name__ == "__main__":
    run_tests()
