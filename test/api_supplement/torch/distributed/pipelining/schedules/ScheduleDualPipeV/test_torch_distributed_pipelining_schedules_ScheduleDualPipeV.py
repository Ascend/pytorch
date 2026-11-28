"""
Test purpose: Verify ScheduleDualPipeV behavior.
API name: torch.distributed.pipelining.schedules.ScheduleDualPipeV
API signature: ScheduleDualPipeV(stages, n_microbatches, loss_fn=None, args_chunk_spec=None,
    kwargs_chunk_spec=None, output_merge_spec=None, scale_grads=True,
    backward_requires_autograd=True, defer_pp_recv=False, max_active_stages=3,
    unshard_lookahead='full', *, defer_reduce_grad_wait=False, max_outstanding_sends=None)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Mirrored stage scheduling | Check mirrored stages and microbatch coverage on two ranks. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- The schedule is constructed and inspected. Training throughput is outside this test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_mirrored_stage_assignment(case, rank, device):
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
    from torch.distributed.pipelining.schedules import (
        analyze_pipeline_activation_liveness,
    )

    case.assertEqual(dict(schedule.stage_index_to_group_rank), {0: 0, 1: 1, 2: 1, 3: 0})
    local_stages = [rank, 3 - rank]
    plan = analyze_pipeline_activation_liveness(
        schedule, pp_rank=rank, stage_indices=local_stages, granularity="stage_microbatch"
    )
    case.assertEqual(
        set(plan.slot_by_stage_and_microbatch),
        {(stage, mb) for stage in local_stages for mb in range(4)},
    )


class TestScheduleDualPipeV(TwoRankTestCase):
    def test_schedule_dual_pipe_v_mirrored_stage_assignment(self):
        self.run_on_two_ranks(check_mirrored_stage_assignment)


if __name__ == "__main__":
    run_tests()
