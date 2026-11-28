"""
Test purpose: Verify register_forward_context behavior.
API name: torch.distributed.pipelining.PipelineStage.register_forward_context
API signature: register_forward_context(self, context_factory)

Coverage dimensions
| Dimension | Covered scenario and expected behavior |
| --- | --- |
| Context registration and removal | Trace each microbatch until the forward hook is removed. |

Execution: Two ranks with HCCL on NPU or NCCL on CUDA.
Scope and prerequisites:
- A singleton subgroup on each rank isolates forward hooks from pipeline transport.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from distributed_test_support import TwoRankTestCase, run_tests

sys.path.pop(0)


def check_context_lifecycle(case, rank, device):
    from contextlib import contextmanager

    import torch
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    subgroups = [dist.new_group([member]) for member in range(2)]
    group = subgroups[rank]
    try:
        sample = torch.ones(1, 2, device=device)
        stage = PipelineStage(
            torch.nn.Identity(), 0, 1, device, input_args=sample, output_args=sample, group=group
        )
        events = []

        @contextmanager
        def capture(info):
            events.append(("enter", info.stage_index, info.microbatch_index))
            yield
            events.append(("exit", info.stage_index, info.microbatch_index))

        handle = stage.register_forward_context(capture)
        schedule = ScheduleGPipe(stage, 2)
        value = torch.arange(4.0, device=device).reshape(2, 2)
        torch.testing.assert_close(schedule.step(value), value)
        case.assertEqual(events, [("enter", 0, 0), ("exit", 0, 0), ("enter", 0, 1), ("exit", 0, 1)])
        handle.remove()
        schedule.step(value)
        case.assertEqual(len(events), 4)
    finally:
        dist.destroy_process_group(group)


class TestRegisterForwardContext(TwoRankTestCase):
    def test_register_forward_context_context_lifecycle(self):
        self.run_on_two_ranks(check_context_lifecycle)


if __name__ == "__main__":
    run_tests()
