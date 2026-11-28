"""
Test purpose: Verify set_reduce_scatter_max_input_buffers behavior.
API name: torch.distributed.fsdp.FSDPModule.set_reduce_scatter_max_input_buffers
API signature: set_reduce_scatter_max_input_buffers(self, max_input_buffers, *, recurse=True)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Buffer count and invalid limits | Reject invalid limits and reduce with one or two buffers. |

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


def check_buffer_limits(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )

    model = sharded_linear(device)
    (group,) = model._get_fsdp_state()._fsdp_param_groups
    with case.assertRaises(ValueError):
        model.set_reduce_scatter_max_input_buffers(0)
    with case.assertRaises(TypeError):
        model.set_reduce_scatter_max_input_buffers(True)
    for count in (1, 2):
        model.set_reduce_scatter_max_input_buffers(count, recurse=False)
        case.assertEqual(group.reduce_scatter_max_input_buffers, count)
        backward_on_rank(model, rank, device)
        check_reduced_gradient(model)


class TestSetReduceScatterMaxInputBuffers(TwoRankTestCase):
    def test_set_reduce_scatter_max_input_buffers_buffer_limits(self):
        self.run_on_two_ranks(check_buffer_limits)


if __name__ == "__main__":
    run_tests()
