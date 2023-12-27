import collections
import contextlib
import warnings

import torch_npu
from . import is_initialized, _get_device_index, _lazy_init

__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "get_allocator_backend"
]


@contextlib.contextmanager
def _free_mutex():
    torch_npu._C._npu_lock_mutex()
    try:
        yield
    finally:
        torch_npu._C._npu_unlock_mutex()


def caching_allocator_alloc(size, device=None, stream=None):
    r"""Performs a memory allocation using the NPU memory allocator.

    Memory is allocated for a given device and a stream, this
    function is intended to be used for interoperability with other
    frameworks. Allocated memory is released through
    :func:`~torch_npu.npu.caching_allocator_delete`.

    Arguments:
        size (int): number of bytes to be allocated.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default NPU device is used.
        stream (torch_npu.npu.Stream or int, optional): selected stream. If is ``None`` then
            the default stream for the selected device is used.

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    if device is None:
        device = torch_npu.npu.current_device()
    device = _get_device_index(device)
    if stream is None:
        stream = torch_npu.npu.current_stream(device)
    if isinstance(stream, torch_npu.npu.streams.Stream):
        stream = stream.npu_stream
    if not isinstance(stream, int):
        raise TypeError('Invalid type for stream argument, must be '
                        '`torch_npu.npu.Stream` or `int` representing a pointer '
                        'to a exisiting stream')
    with torch_npu.npu.device(device):
        return torch_npu._C._npu_npuCachingAllocator_raw_alloc(size, stream)


def caching_allocator_delete(mem_ptr):
    r"""Deletes memory allocated using the NPU memory allocator.

    Memory allocated with :func:`~torch_npu.npu.caching_allocator_alloc`.
    is freed here. The associated device and stream are tracked inside
    the allocator.

    Arguments:
        mem_ptr (int): memory address to be freed by the allocator.

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    torch_npu._C._npu_npuCachingAllocator_raw_delete(mem_ptr)


def set_per_process_memory_fraction(fraction, device=None) -> None:
    r"""Set memory fraction for a process.
    The fraction is used to limit an caching allocator to allocated memory on a NPU device.
    The allowed value equals the total visible memory multiplied fraction.
    If trying to allocate more than the allowed value in a process, will raise an out of
    memory error in allocator.
    Arguments:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (torch.device or int, optional): selected device. If it is
            ``None`` the default NPU device is used.
    .. note::
        In general, the total available free memory is less than the total capacity.
    """
    _lazy_init()
    if device is None:
        device = torch_npu.npu.current_device()
    device = _get_device_index(device)
    if not isinstance(fraction, float):
        raise TypeError('Invalid type for fraction argument, must be `float`')
    if fraction < 0 or fraction > 1:
        raise ValueError('Invalid fraction value: {}. '
                         'Allowed range: 0~1'.format(fraction))

    torch_npu._C._npu_setMemoryFraction(fraction, device)


def empty_cache():
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other NPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch_npu.npu.empty_cache` doesn't increase the amount of NPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of NPU memory in certain cases. See :ref:`npu-memory-management` for
        more details about NPU memory management.
    """
    if is_initialized():
        torch_npu._C._npu_emptyCache()


def memory_stats(device=None):
    """Returns a dictionary of NPU memory allocator statistics for a
    given device.
    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.
    Core statistics:
    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from ``npuMalloc()``.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.
    For these core statistics, values are broken down as follows.
    Pool type:
    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of October 2019, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of October 2019, for size < 1MB allocations).
    Metric type:
    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.
    In addition to the core statistics, we also provide some simple event
    counters:
    - ``"num_alloc_retries"``: number of failed ``npuMalloc`` calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    The caching allocator can be configured via ENV to not split blocks larger than a
    defined size (see Memory Management section of the Cuda Semantics documentation).
    This helps avoid memory framentation but may have a performance
    penalty. Additional outputs to assist with tuning and evaluating impact:
    - ``"max_split_size"``: blocks above this size will not be split.
    - ``"oversize_allocations.{current,peak,allocated,freed}"``:
      number of over-size allocation requests received by the memory allocator.
    - ``"oversize_segments.{current,peak,allocated,freed}"``:
      number of over-size reserved segments from ``cudaMalloc()``.
    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistics for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).
    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    result = []

    def _recurse_add_to_result(prefix, obj):
        if isinstance(obj, dict):
            if len(prefix) > 0:
                prefix += "."
            for k, v in obj.items():
                _recurse_add_to_result(prefix + k, v)
        else:
            result.append((prefix, obj))

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result("", stats)
    result.sort()

    return collections.OrderedDict(result)


def memory_stats_as_nested_dict(device=None):
    r"""Returns the result of :func:`~torch_npu.npu.memory_stats` as a nested dictionary."""
    device = _get_device_index(device, optional=True)
    return torch_npu._C._npu_memoryStats(device)


def reset_accumulated_memory_stats(device=None):
    r"""Resets the "accumulated" (historical) stats tracked by the NPU memory allocator.

    See :func:`~torch_npu.npu.memory_stats` for details. Accumulated stats correspond to
    the `"allocated"` and `"freed"` keys in each individual stat dict, as well as
    `"num_alloc_retries"` and `"num_ooms"`.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch_npu._C._npu_resetAccumulatedMemoryStats(device)


def reset_peak_memory_stats(device=None):
    r"""Resets the "peak" stats tracked by the NPU memory allocator.

    See :func:`~torch_npu.npu.memory_stats` for details. Peak stats correspond to the
    `"peak"` key in each individual stat dict.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    return torch_npu._C._npu_resetPeakMemoryStats(device)


def reset_max_memory_allocated(device=None):
    r"""Resets the starting point in tracking maximum NPU memory occupied by
    tensors for a given device.

    See :func:`~torch_npu.npu.max_memory_allocated` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch_npu.npu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    warnings.warn(
        "torch_npu.npu.reset_max_memory_allocated now calls torch_npu.npu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        DeprecationWarning)
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device=None):
    r"""Resets the starting point in tracking maximum NPU memory managed by the
    caching allocator for a given device.

    See :func:`~torch_npu.npu.max_memory_cached` for details.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. warning::
        This function now calls :func:`~torch_npu.npu.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    warnings.warn(
        "torch_npu.npu.reset_max_memory_cached now calls torch_npu.npu.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        DeprecationWarning)
    return reset_peak_memory_stats(device=device)


def memory_allocated(device=None):
    r"""Returns the current NPU memory occupied by tensors in bytes for a given
    device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on NPU. See :ref:`npu-memory-management` for more
        details about NPU memory management.
    """
    return memory_stats(device=device)["allocated_bytes.all.current"]


def max_memory_allocated(device=None):
    r"""Returns the maximum NPU memory occupied by tensors in bytes for a given
    device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch_npu.npu.reset_peak_stats` can be used to
    reset the starting point in tracking this metric. For example, these two
    functions can measure the peak allocated memory usage of each iteration in a
    training loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    return memory_stats(device=device)["allocated_bytes.all.peak"]


def memory_reserved(device=None):
    r"""Returns the current NPU memory managed by the caching allocator in bytes
    for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    return memory_stats(device=device)["reserved_bytes.all.current"]


def max_memory_reserved(device=None):
    r"""Returns the maximum NPU memory managed by the caching allocator in bytes
    for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch_npu.npu.reset_peak_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    return memory_stats(device=device)["reserved_bytes.all.peak"]


def memory_cached(device=None):
    r"""Deprecated; see :func:`~torch_npu.npu.memory_reserved`."""
    warnings.warn(
        "torch_npu.npu.memory_cached has been renamed to torch_npu.npu.memory_reserved",
        DeprecationWarning)
    return memory_reserved(device=device)


def max_memory_cached(device=None):
    r"""Deprecated; see :func:`~torch_npu.npu.max_memory_reserved`."""
    warnings.warn(
        "torch_npu.npu.max_memory_cached has been renamed to torch_npu.npu.max_memory_reserved",
        DeprecationWarning)
    return max_memory_reserved(device=device)


def memory_snapshot():
    r"""Returns a snapshot of the NPU memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals.

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    return torch_npu._C._npu_memorySnapshot()


def _format_size(sz, pref_sz):
    prefixes = ["B ", "KB", "MB", "GB", "TB", "PB"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_sz < 768 * 1024:
            break
        prefix = new_prefix
        sz //= 1024
        pref_sz /= 1024
    return "{:7d} {}".format(sz, prefix)


def _format_count(cnt, pref_cnt):
    prefixes = [" ", "K", "M"]
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_cnt < 750 * 1000:
            break
        prefix = new_prefix
        cnt //= 1000
        pref_cnt /= 1000
    return "{:7d} {} ".format(cnt, prefix)


def create_metrics_to_display():
    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("reserved_bytes", "NPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "NPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} PyTorch NPU memory summary, device ID {device:<18d} ")
    lines.append("-" * 75)
    lines.append("  {_:9} NPU OOMs: {num_ooms:<13d} | {_:6} npuMalloc retries: {num_alloc_retries:<9d}  ")
    lines.append("=" * 75)
    lines.append("        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  ")
    return metrics_to_display, lines


def memory_summary(device=None, abbreviated=False):
    r"""Returns a human-readable printout of the current memory allocator
    statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).

    .. note::
        See :ref:`npu-memory-management` for more details about NPU memory
        management.
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)
    metrics_to_display, lines = create_metrics_to_display()

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = None, None, None, None

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(" {:<21} | {} | {} | {} | {} ".format(
                submetric_name,
                formatter(current, current_prefval),
                formatter(peak, peak_prefval),
                formatter(allocated, allocated_prefval),
                formatter(freed, freed_prefval)), )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize NPU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(" {:<21} | {} | {} | {} | {} ".format(
            metric_name,
            formatter(current, current),
            formatter(peak, peak),
            formatter(allocated, allocated),
            formatter(freed, freed)), )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def get_allocator_backend() -> str:
    r"""Returns a string describing the active allocator backend as set by
    ``PYTORCH_NPU_ALLOC_CONF``. Currently available backends are
    ``native`` (PyTorch's native caching allocator).

    .. note::
        See :ref:`npu-memory-management` for details on choosing the allocator backend.
    """
    return torch_npu._C._npu_getAllocatorBackend()
