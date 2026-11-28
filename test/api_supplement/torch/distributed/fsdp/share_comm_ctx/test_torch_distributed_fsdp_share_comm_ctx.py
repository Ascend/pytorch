"""
Test purpose: Verify share_comm_ctx behavior.
API name: torch.distributed.fsdp.share_comm_ctx
API signature: share_comm_ctx(modules)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Communication context sharing | Share a context across two sharded roots and check their gradients. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The fixture uses two accelerator ranks and one sharded model topology. Mixed-device training is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_independent_roots(case, rank, device):
    from distributed_test_support import (
        backward_on_rank,
        check_reduced_gradient,
        sharded_linear,
    )
    from torch.distributed.fsdp import share_comm_ctx

    roots = [sharded_linear(device), sharded_linear(device)]
    for root in roots:
        root._get_fsdp_state()._lazy_init()
    share_comm_ctx(roots)
    state_a, state_b = [root._get_fsdp_state() for root in roots]
    case.assertIs(state_a._comm_ctx, state_b._comm_ctx)
    for root in roots:
        backward_on_rank(root, rank, device)
        check_reduced_gradient(root)


class TestShareCommCtx(TwoRankTestCase):
    def test_share_comm_ctx_independent_roots(self):
        self.run_on_two_ranks(check_independent_roots)


if __name__ == "__main__":
    run_tests()
