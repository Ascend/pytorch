from enum import Enum

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

    def __call__(self, step: int) -> ProfilerAction:
        if step == CLOSE_STEP:
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


def default_schedule_fn(step: int) -> ProfilerAction:
    if step == CLOSE_STEP:
        return ProfilerAction.NONE
    return ProfilerAction.RECORD_AND_SAVE
