# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a Schedule zoo for testing torch.distributed.pipelining.
# It includes schedules designed purely for testing purposes
from typing import Callable, Optional

from torch.distributed.pipelining.schedules import (
    _Action,
    _ComputationType,
    _PipelineScheduleRuntime,
    PipelineScheduleMulti,
    RECV_B,
    RECV_F,
    SEND_B,
    SEND_F,
)
from torch.distributed.pipelining.stage import _PipelineStageBase


F = _ComputationType.FORWARD
B = _ComputationType.FULL_BACKWARD
W = _ComputationType.BACKWARD_WEIGHT
INPUT = _ComputationType.BACKWARD_INPUT


class ScheduleVShaped(PipelineScheduleMulti):
    n_stages = 4
    rank_stages = {
        0: [0, 3],
        1: [1, 2],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Go through one microbatch
        # Note(whc) - it might be easier to work with these schedules by writing them as a list of
        # ["0F0", ...] and then parsing them in the test infra to turn them into actions.
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                None,
                None,
                _Action(3, F, 0),
                _Action(3, B, 0),
                None,
                None,
                _Action(0, B, 0),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(2, F, 0),
                None,
                None,
                _Action(2, B, 0),
                _Action(1, B, 0),
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleUnbalanced(PipelineScheduleMulti):
    n_stages = 5
    rank_stages = {
        0: [0, 1, 4],
        1: [2, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(1, F, 0),
                None,
                None,
                _Action(4, F, 0),
                _Action(4, B, 0),
                None,
                None,
                _Action(1, B, 0),
                _Action(0, B, 0),
            ],
            1: [
                None,
                None,
                _Action(2, F, 0),
                _Action(3, F, 0),
                None,
                None,
                _Action(3, B, 0),
                _Action(2, B, 0),
                None,
                None,
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithW(PipelineScheduleMulti):
    n_stages = 4
    num_microbatches = 2
    rank_stages = {
        0: [0, 2],
        1: [1, 3],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        enable_zero_bubble: bool = True,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )

        # Needs to be updated as part of all schedules using "W"
        self.use_full_backward = False

        # Go through two microbatches
        self.pipeline_order = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(2, F, 0),
                _Action(2, F, 1),
                None,
                _Action(2, INPUT, 0),
                _Action(2, W, 0),
                _Action(0, INPUT, 0),
                _Action(2, INPUT, 1),
                _Action(0, W, 0),
                _Action(0, INPUT, 1),
                _Action(2, W, 1),
                _Action(0, W, 1),
            ],
            1: [
                None,
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(3, F, 0),
                _Action(3, INPUT, 0),
                _Action(3, F, 1),
                _Action(1, INPUT, 0),
                _Action(3, INPUT, 1),
                _Action(3, W, 0),
                _Action(1, INPUT, 1),
                _Action(1, W, 0),
                _Action(3, W, 1),
                _Action(1, W, 1),
            ],
        }
        self._validate_and_set_stage_mapping(self.pipeline_order)


class ScheduleWithReorderedB(_PipelineScheduleRuntime):
    n_stages = 2
    num_microbatches = 2
    rank_stages = {
        0: [0],
        1: [1],
    }

    def __init__(
        self,
        stages: list[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        scale_grads: bool = True,
    ):
        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            scale_grads=scale_grads,
        )
        # Go through two microbatches
        actions: dict[int, list[_Action | None]] = {
            0: [
                _Action(0, F, 0),
                _Action(0, F, 1),
                _Action(0, SEND_F, 0),
                _Action(0, SEND_F, 1),
            ],
            1: [
                _Action(1, RECV_F, 0),
                _Action(1, RECV_F, 1),
                _Action(1, F, 0),
                _Action(1, F, 1),
                _Action(1, B, 0),
                _Action(1, B, 1),
                _Action(1, SEND_B, 0),
                _Action(1, SEND_B, 1),
            ],
        }
        self._add_wait_send_actions(actions)
        actions[0] += [
            _Action(0, RECV_B, 0),
            _Action(0, RECV_B, 1),
            _Action(0, B, 0),
            _Action(0, B, 1),
        ]
        self._prepare_schedule_with_comms(actions, format="compute_comms")

    def _add_wait_send_actions(self, actions: dict[int, list[_Action | None]]) -> None:
        """Hook for version-specific WAIT_SEND actions; no-op in this schedule."""


class ScheduleWithReorderedBWithWaits(ScheduleWithReorderedB):
    """ScheduleWithReorderedB variant for torch >= 2.15.

    torch >= 2.15 tracks in-flight sends and requires explicit WAIT_SEND actions;
    this variant is used by the matching version-gated test case.
    """

    def _add_wait_send_actions(self, actions: dict[int, list[_Action | None]]) -> None:
        # Imported lazily: WAIT_SEND actions do not exist on torch < 2.15, where
        # this class is never instantiated (see the runIfVersion-gated test).
        from torch.distributed.pipelining.schedules import (  # type: ignore[attr-defined]
            WAIT_SEND_B,
            WAIT_SEND_F,
        )

        actions[0] += [
            _Action(0, WAIT_SEND_F, 0),
            _Action(0, WAIT_SEND_F, 1),
        ]
        actions[1] += [
            _Action(1, WAIT_SEND_B, 0),
            _Action(1, WAIT_SEND_B, 1),
        ]
