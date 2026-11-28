"""
Test purpose: Verify _time_estimator behavior.
API name: torch.distributed.distributed_c10d._time_estimator
API signature: _time_estimator(group=None, device=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Time estimation capability | Check the estimate or the unsupported-backend error. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Positive timing estimates require a backend that advertises support. Wall-clock performance is
  outside this case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_estimation_capability(case, rank, device):
    import torch
    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _time_estimator

    value = torch.tensor([rank + 1], device=device)
    dist.all_reduce(value)
    backend = dist.group.WORLD._get_backend(device)
    if backend._supports_time_estimate:
        with _time_estimator(group=dist.group.WORLD, device=device) as estimate:
            dist.all_reduce(value)
        case.assertGreaterEqual(estimate.estimated_time, 0)
    else:
        with case.assertRaisesRegex(NotImplementedError, "not supported"):
            with _time_estimator(device=device):
                case.fail("Unsupported backend entered estimation mode")


class TestTimeEstimator(TwoRankTestCase):
    def test_time_estimator_estimation_capability(self):
        self.run_on_two_ranks(check_estimation_capability)


if __name__ == "__main__":
    run_tests()
