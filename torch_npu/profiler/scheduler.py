from enum import Enum

from torch_npu.utils._error_code import ErrCode, prof_error
from .analysis.prof_common_func._constant import print_warn_msg

__all__ = [
    'ProfilerAction',
    'Schedule'
]


class ProfilerAction(Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


class Schedule:
    """
    The profiler will skip the first ``skip_first`` steps, then wait for ``wait`` steps,
    then do the warmup for the next ``warmup`` steps, then do the active recording for the next
    ``active`` steps and then repeat the cycle starting with ``wait`` steps. The optional number
    of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.
    """
    def __init__(self, wait: int, active: int, warmup: int = 0, repeat: int = 0, skip_first: int = 0) -> None:
        self.wait = wait
        self.active = active
        self.warmup = warmup
        self.repeat = repeat
        self.skip_first = skip_first
        self._check_params()

    def __call__(self, step: int) -> ProfilerAction:
        if step < 0:
            raise ValueError("Invalid parameter step, which must be not less than 0." + prof_error(ErrCode.VALUE))
        if step < self.skip_first:
            return ProfilerAction.NONE
        else:
            step -= self.skip_first
        num_steps = self.wait + self.warmup + self.active
        if self.repeat > 0 and step / num_steps >= self.repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < self.wait:
            return ProfilerAction.NONE
        elif mod_step < self.wait + self.warmup:
            return ProfilerAction.WARMUP
        else:
            return (
                ProfilerAction.RECORD
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE
            )

    def _check_params(self):
        if not isinstance(self.wait, int) or self.wait < 0:
            print_warn_msg("Invalid parameter wait, reset it to 0.")
            self.wait = 0
        if not isinstance(self.warmup, int) or self.warmup < 0:
            print_warn_msg("Invalid parameter warmup, reset it to 0.")
            self.warmup = 0
        if not isinstance(self.active, int) or self.active <= 0:
            print_warn_msg("Invalid parameter active, reset it to 1.")
            self.active = 1
        if not isinstance(self.repeat, int) or self.repeat < 0:
            print_warn_msg("Invalid parameter repeat, reset it to 0.")
            self.repeat = 0
        if not isinstance(self.skip_first, int) or self.skip_first < 0:
            print_warn_msg("Invalid parameter skip_first, reset it to 0.")
            self.skip_first = 0

        if self.warmup == 0:
            print_warn_msg("Profiler won't be using warmup, this can skew profiler results")


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD
