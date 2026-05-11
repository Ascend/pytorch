__all__ = []


import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error


def _set_thread_affinity(core_range: list[int] | list[list[int]] | None = None):
    """Set thread CPU affinity.

    Args:
        core_range: CPU core range to bind to, supports three forms:
            - None: reset to default affinity.
            - [start, end]: single range, binds to cores from start to end (inclusive).
              Example: [0, 3] binds to CPU cores 0, 1, 2, 3.
            - [[start1, end1], [start2, end2], ...]: multiple ranges.
              Example: [[0, 3], [8, 11]] binds to CPU cores 0, 1, 2, 3, 8, 9, 10, 11.
    """
    if core_range is None:
        torch_npu._C._npu_set_thread_affinity(-1, -1)
        return

    # Single range: [start, end]
    if (
        isinstance(core_range, list)
        and len(core_range) == 2
        and all(isinstance(x, int) for x in core_range)
    ):
        core_ranges = [core_range]
    # Multiple ranges: [[start1, end1], [start2, end2], ...]
    elif (
        isinstance(core_range, list)
        and len(core_range) > 0
        and all(isinstance(x, list) and len(x) == 2 for x in core_range)
    ):
        core_ranges = core_range
    else:
        raise ValueError(f"Invalid core range: {core_range}" + pta_error(ErrCode.PARAM))

    for start, end in core_ranges:
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and start >= 0
            and start <= end
        ):
            raise ValueError(
                f"Invalid core range values: [{start}, {end}]"
                + pta_error(ErrCode.VALUE)
            )

    core_list = [core for start, end in core_ranges for core in range(start, end + 1)]
    core_list = sorted(set(core_list))
    torch_npu._C._npu_set_thread_affinity(core_list)


def _reset_thread_affinity():
    torch_npu._C._npu_reset_thread_affinity()
