"""
Test purpose: Verify set_custom_reduce_scatter behavior.
API name: torch.distributed.fsdp.FSDPModule.set_custom_reduce_scatter
API signature: set_custom_reduce_scatter(self, comm)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Custom reduce-scatter replacement | Replace the collector and check which one runs during training. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_custom_collective(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        DefaultReduceScatter,
    )

    class RecordingReduceScatter(DefaultReduceScatter):
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return super().__call__(*args, **kwargs)

    first, second = RecordingReduceScatter(), RecordingReduceScatter()
    model.set_custom_reduce_scatter(first)
    model.set_custom_reduce_scatter(second)
    backward_on_rank(model, rank, device)
    check_reduced_gradient(model)
    case.assertEqual(first.calls, 0)
    case.assertGreater(second.calls, 0)


class TestSetCustomReduceScatter(TwoRankTestCase):
    def test_set_custom_reduce_scatter_custom_collective(self):
        self.run_on_two_ranks(check_custom_collective)


if __name__ == "__main__":
    run_tests()
