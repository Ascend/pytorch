"""
Test purpose: Verify analyze_pipeline_activation_liveness behavior.
API name: torch.distributed.pipelining.schedules.analyze_pipeline_activation_liveness
API signature: analyze_pipeline_activation_liveness(schedule, *, pp_rank, stage_indices,
    granularity)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Overlapping activation lifetimes | Place overlapping activations in distinct slots. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- Both granularities are exercised with four microbatches; numeric model training is outside
  liveness analysis.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_overlapping_activations(case, rank, device):
    import torch
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import ScheduleDualPipeV

    sample = torch.ones(1, 2, device=device)
    stages = [
        PipelineStage(
            torch.nn.Linear(2, 2, device=device),
            index,
            4,
            device,
            input_args=sample,
            output_args=sample,
        )
        for index in (rank, 3 - rank)
    ]
    schedule = ScheduleDualPipeV(stages, n_microbatches=4)
    from itertools import combinations

    from torch.distributed.pipelining.schedules import (
        analyze_pipeline_activation_liveness,
    )

    for granularity in ("stage_microbatch", "microbatch"):
        plan = analyze_pipeline_activation_liveness(
            schedule, pp_rank=rank, stage_indices=(rank, 3 - rank), granularity=granularity
        )
        keys = [(rank, mb) for mb in range(4)]
        for left, right in combinations(keys, 2):
            start_a, end_a = plan.get_activation_lifetime(*left)
            start_b, end_b = plan.get_activation_lifetime(*right)
            if max(start_a, start_b) <= min(end_a, end_b):
                case.assertNotEqual(plan.slot_for(*left), plan.slot_for(*right))
    with case.assertRaisesRegex(ValueError, "empty"):
        analyze_pipeline_activation_liveness(
            schedule, pp_rank=rank, stage_indices=(), granularity="microbatch"
        )


class TestAnalyzePipelineActivationLiveness(TwoRankTestCase):
    def test_analyze_pipeline_activation_liveness_overlapping_activations(self):
        self.run_on_two_ranks(check_overlapping_activations)


if __name__ == "__main__":
    run_tests()
