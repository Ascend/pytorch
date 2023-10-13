from enum import Enum

from .analysis.prof_common_func.constant import print_warn_msg

CLOSE_STEP = -99


class ProfilerAction(Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


class Schedule:
    def __init__(self, wait: int, active: int, warmup: int = 0, repeat: int = 0, skip_first: int = 0) -> any:
        self._wait = wait
        self._warmup = warmup
        self._active = active
        self._repeat = repeat
        self._skip_first = skip_first
        self._check_params()

    def __call__(self, step: int) -> ProfilerAction:
        if step == CLOSE_STEP:
            return ProfilerAction.NONE
        if self._active == 0:
            return ProfilerAction.NONE
        if step < self._skip_first:
            return ProfilerAction.NONE
        else:
            step -= self._skip_first
        num_steps = self._wait + self._warmup + self._active
        if self._repeat > 0 and step / num_steps >= self._repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < self._wait:
            return ProfilerAction.NONE
        elif mod_step < self._wait + self._warmup:
            return ProfilerAction.WARMUP
        else:
            return ProfilerAction.RECORD if mod_step < num_steps - 1 else ProfilerAction.RECORD_AND_SAVE

    def _check_params(self):
        try:
            self._wait = int(self._wait)
            if self._wait < 0:
                raise ValueError
        except ValueError:
            print_warn_msg(
                "Invalid parameter wait, which must be an integer greater than or equal to 0, reset it to 0.")
            self._wait = 0

        try:
            self._warmup = int(self._warmup)
            if self._warmup < 0:
                raise ValueError
        except ValueError:
            print_warn_msg(
                "Invalid parameter warmup, which must be an integer greater than or equal to 0, reset it to 0.")
            self._warmup = 0

        try:
            self._active = int(self._active)
            if self._active < 0:
                raise ValueError
        except ValueError:
            print_warn_msg(
                "Invalid parameter active, which must be an integer greater than or equal to 0, reset it to 0.")
            self._active = 0

        try:
            self._repeat = int(self._repeat)
            if self._repeat < 0:
                raise ValueError
        except ValueError:
            print_warn_msg(
                "Invalid parameter repeat, which must be an integer greater than or equal to 0, reset it to 0.")
            self._repeat = 0

        try:
            self._skip_first = int(self._skip_first)
            if self._skip_first < 0:
                raise ValueError
        except ValueError:
            print_warn_msg(
                "Invalid parameter skip_first, which must be an integer greater than or equal to 0, reset it to 0.")
            self._skip_first = 0


def default_schedule_fn(step: int) -> ProfilerAction:
    if step == CLOSE_STEP:
        return ProfilerAction.NONE
    return ProfilerAction.RECORD_AND_SAVE
