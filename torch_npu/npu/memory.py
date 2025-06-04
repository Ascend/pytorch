import collections
import contextlib
import warnings
import ctypes
import pickle
import sys
import os
import stat
import platform
from typing import Any, Dict, Optional, Tuple, Union

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error
from . import is_initialized, _get_device_index, _lazy_init
from .utils import _dummy_type
from ._memory_viz import memory as _memory, segments as _segments

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
    "get_allocator_backend",
    "NPUPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "MemPoolContext",
    "use_mem_pool",
]

if not hasattr(torch_npu._C, "_npu_NPUAllocator"):
    # Define dummy base classes
    torch_npu._C.__dict__["_npu_NPUAllocator"] = _dummy_type("_npu_NPUAllocator")

if not hasattr(torch_npu._C, "_MemPool"):
    # Define dummy base classes
    torch_npu._C.__dict__["_MemPool"] = _dummy_type("_MemPool")
    torch_npu._C.__dict__["_MemPoolContext"] = _dummy_type("_MemPoolContext")
    torch_npu._C.__dict__["_npu_beginAllocateToPool"] = _dummy_type(
        "_npu_beginAllocateToPool"
    )
    torch_npu._C.__dict__["_npu_endAllocateCurrentStreamToPool"] = _dummy_type(
        "_npu_endAllocateCurrentStreamToPool"
    )

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
                        'to a exisiting stream' + pta_error(ErrCode.TYPE))
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
        raise TypeError('Invalid type for fraction argument, must be `float`' + pta_error(ErrCode.TYPE))
    if fraction < 0 or fraction > 1:
        raise ValueError('Invalid fraction value: {}. '
                         'Allowed range: 0~1'.format(fraction) + pta_error(ErrCode.VALUE))

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
    if not is_initialized():
        return {}
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
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


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
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


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
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


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
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


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
    return torch_npu._C._npu_memorySnapshot()["segments"]


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


def _create_metrics_to_display():
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
    metrics_to_display, lines = _create_metrics_to_display()

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


class _NPUAllocator:
    r"""Wrapper over internal NPU memory allocators."""

    def __init__(self, allocator: torch_npu._C._npu_NPUAllocator):
        self._allocator = allocator

    def allocator(self):
        return self._allocator


class NPUPluggableAllocator(_NPUAllocator):
    r"""NPU memory allocator loaded from a so file."""

    def __init__(self, path_to_so_file: str, alloc_fn_name: str, free_fn_name: str):
        r"""Memory allocators are compiled in .so files and loaded dynamically using ctypes.

        To change the active allocator use the :func:`torch_npu.npu.change_current_allocator` function.

        Args:
            path_to_so_file(str): Path in the filesystem to the `.so` file containing
                the allocator functions
            alloc_fn_name(str): Name of the function to perform the memory allocation
                in the so file. The signature must be:
                void* alloc_fn_name(ssize_t size, int device, aclrtStream stream);
            free_fn_name(str): Name of the function to perform the memory release
                in the so file. The signature must be:
                void free_fn_name(void* ptr, size_t size, aclrtStream stream);

        .. warning::
            This is currently supported only in unix OSs

        .. note::
            See :ref:`npu-memory-management` for details on creating and using a custom allocator
        """
        allocator = ctypes.CDLL(path_to_so_file)
        alloc_fn = ctypes.cast(getattr(allocator, alloc_fn_name), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(allocator, free_fn_name), ctypes.c_void_p).value
        if alloc_fn is None:
            raise RuntimeError('alloc_fn is None' + pta_error(ErrCode.NOT_FOUND))
        if free_fn is None:
            raise RuntimeError('free_fn is None' + pta_error(ErrCode.NOT_FOUND))
        self._allocator = torch_npu._C._npu_customAllocator(alloc_fn, free_fn)


def change_current_allocator(allocator: _NPUAllocator) -> None:
    r"""Change the currently used memory allocator to be the one provided.

    If the current allocator has already been used/initialized, this function will error.


    Args:
        allocator (torch_npu.npu.memory._NPUAllocator): allocator to be set as the active one.
    .. note::
        See :ref:`npu-memory-management` for details on creating and using a custom allocator
    """
    torch_npu._C._npu_changeCurrentAllocator(allocator.allocator())


def _get_current_allocator() -> _NPUAllocator:
    r"""Return the allocator being currently used.

    .. note::
        See :ref:`npu-memory-management` for details on creating and using a custom allocator
    """
    return _NPUAllocator(torch_npu._C._npu_getAllocator())


class MemPool(torch_npu._C._MemPool):
    r"""MemPool represents a pool of memory in a caching allocator. Currently,
    it's just the ID of the pool object maintained in the NPUCachingAllocator.

    Args:
        allocator(torch_npu._C._npu_NPUAllocator, optional): a
            torch_npu._C._npu_NPUAllocator object that can be used to
            define how memory gets allocated in the pool. If :attr:`allocator`
            is ``None`` (default), memory allocation follows the default/
            current configuration of the NPUCachingAllocator.

    """

    def __init__(self, allocator: Optional[torch_npu._C._npu_NPUAllocator] = None):
        super().__init__(allocator, True)

    @property
    def id(self) -> Tuple[int, int]:
        r"""Returns the ID of this pool as a tuple of two ints."""
        return super().id

    @property
    def allocator(self) -> Optional[torch_npu._C._npu_NPUAllocator]:
        r"""Returns the allocator this MemPool routes allocations to"""
        return super().allocator


class MemPoolContext(torch_npu._C._MemPoolContext):
    r"""MemPoolContext holds the currently active pool and stashes the previous
    pool. On deletion it makes the previous pool active.

    Args:
        pool(torch_npu.npu.MemPool): a MemPool object to be made active so that
        allocations route to this pool.

    """

    def __init__(self, pool: MemPool):
        super().__init__(pool)

    @staticmethod
    def active_pool() -> Optional[torch_npu._C._MemPool]:
        r"""Returns the active MemPool"""
        return torch_npu._C._MemPoolContext.active_pool()


@contextlib.contextmanager
def use_mem_pool(pool: MemPool, device=None):
    r"""A context manager that routes allocations to a given pool.

    Args:
        pool(torch_npu.npu.MemPool): a MemPool object to be made active so that
            allocations route to this pool.
        device (torch.device or int, optional): selected device. Uses MemPool on
            the current device, given by :func:`~torch_npu.npu.current_device,
            if :attr:`device` is ``None`` (default).

    """
    ctx = MemPoolContext(pool)
    device_index = (
        torch_npu.npu.current_device() if device is None else _get_device_index(device)
    )
    torch_npu._C._npu_beginAllocateToPool(device_index, pool.id)
    try:
        yield
    finally:
        torch_npu._C._npu_endAllocateCurrentStreamToPool(device_index, pool.id)
        del ctx


def _record_memory_history(enabled="all", *args, **kwargs):
    """Enable recording of stack traces associated with memory
    allocations, so you can tell what allocated any piece of memory in
    :func:`torch.npu.memory._snapshot()`.

    In addition too keeping stack traces with each current allocation and free,
    this will also enable recording of a history of all alloc/free events.

    Use :func:`torch.npu.memory._snapshot()` to retrieve this information,
    and the tools in `_memory_viz.py` to visualize snapshots.

    The Python trace collection is fast (2us per trace), so you may consider
    enabling this on production jobs if you anticipate ever having to debug
    memory issues.

    C++ trace collection is also fast (~50ns/frame), which for many typical programs
    works out to ~2us per trace, but can vary depending on stack depth.

    Args:
        enabled (Literal[None, "state", "all"], optional):
            `None`, disable recording memory history.
            `"state"`, keep information for currenly allocated memory.
            `"all"`, additionally keep a history of all alloc/free calls.
            Defaults to "all".
        context (Literal[None, "state", "alloc", "all"], optional):
            `None`, Do not record any tracebacks.
            `"state"`, Record tracebacks for currently allocated memory.
            `"alloc"`, additionally keep tracebacks for alloc calls.
            `"all"`, additionally keep tracebacks for free calls.
            Defaults to "all".
        stacks (Literal["python", "all"], optional):
            `"python"`, include Python, TorchScript, and inductor frames in tracebacks
            `"all"`, additionally include C++ frames
            Defaults to "all".
        max_entries (int, optional): Keep a maximum of `max_entries`
            alloc/free events in the recorded history recorded.
    """
    return _record_memory_history_impl(enabled, *args, **kwargs)


def _record_memory_history_impl(
    enabled: Optional[str] = "all",
    context: Optional[str] = "all",
    stacks: str = "all",
    max_entries: int = sys.maxsize,
    device=None,
):
    torch_npu.npu._lazy_init()
    torch_npu._C._npu_record_memory_history(enabled, context, stacks, max_entries)


def _snapshot(device=None):
    """Save a snapshot of NPU memory state at the time it was called.

    The state is represented as a dictionary with the following structure.

    .. code-block:: python

        class Snapshot(TypedDict):
            segments : List[Segment]
            device_traces: List[List[TraceEntry]]

        class Segment(TypedDict):
            # Segments are memory returned from a aclrtMalloc call.
            # The size of reserved memory is the sum of all Segments.
            # Segments are cached and reused for future allocations.
            # If the reuse is smaller than the segment, the segment
            # is split into more then one Block.
            # empty_cache() frees Segments that are entirely inactive.
            address: int
            total_size: int #  aclrtMalloc'd size of segment
            stream: int
            segment_type: Literal['small', 'large'] # 'large' (>1MB)
            allocated_size: int # size of memory in use
            active_size: int # size of memory in use or in active_awaiting_free state
            blocks : List[Block]

        class Block(TypedDict):
            # A piece of memory returned from the allocator, or
            # current cached but inactive.
            size: int
            requested_size: int # size requested during malloc, may be smaller than
                                # size due to rounding
            address: int
            state: Literal['active_allocated', # used by a tensor
                        'active_awaiting_free', # waiting for another stream to finish using
                                                # this, then it will become free
                        'inactive',] # free for reuse
            frames: List[Frame] # stack trace from where the allocation occurred

        class Frame(TypedDict):
                filename: str
                line: int
                name: str

        class TraceEntry(TypedDict):
            # When `torch.npu.memory._record_memory_history()` is enabled,
            # the snapshot will contain TraceEntry objects that record each
            # action the allocator took.
            action: Literal[
            'alloc'  # memory allocated
            'free_requested', # the allocated received a call to free memory
            'free_completed', # the memory that was requested to be freed is now
                            # able to be used in future allocation calls
            'segment_alloc', # the caching allocator ask aclrtMalloc for more memory
                            # and added it as a segment in its cache
            'segment_free',  # the caching allocator called aclrtFree to return memory
                            # to npu possibly trying free up memory to
                            # allocate more segments or because empty_caches was called
            'oom',          # the allocator threw an OOM exception. 'size' is
                            # the requested number of bytes that did not succeed
            'snapshot'      # the allocator generated a memory snapshot
                            # useful to coorelate a previously taken
                            # snapshot with this trace
            ]
            addr: int # not present for OOM
            frames: List[Frame]
            size: int
            stream: int
            device_free: int # only present for OOM, the amount of
                            # memory npu still reports to be free

    Returns:
        The Snapshot dictionary object
    """
    return torch_npu._C._npu_memorySnapshot()


def _dump_snapshot(filename="dump_snapshot.pickle"):
    """
    Save a pickled version of the `torch.memory._snapshot()` dictionary to a file.

    This file can be opened by the interactive snapshot viewer at pytorch.org/memory_viz

    Args:
        filename (str, optional): Name of the file to create. Defaults to "dump_snapshot.pickle".
    """
    s = _snapshot()
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR), "wb") as f:
        pickle.dump(s, f)

    prof_path = os.path.dirname(os.path.abspath(filename))
    activities = {torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU}
    torch_npu._C._profiler._init_profiler(prof_path, activities)
    prof_config = [prof_path, False, True, False, False, False, torch_npu._C._profiler._ExperimentalConfig()]
    npu_prof_config = torch_npu._C._profiler.NpuProfilerConfig(*tuple(prof_config))
    torch_npu._C._profiler._start_profiler(npu_prof_config, activities)

    torch_npu._C._profiler._stop_profiler()
    torch_npu._C._profiler._finalize_profiler()


def _save_segment_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR), "w") as f:
        f.write(_segments(snapshot))


def _save_memory_usage(filename="output.svg", snapshot=None):
    if snapshot is None:
        snapshot = _snapshot()
    with os.fdopen(os.open(filename, os.O_WRONLY | os.O_CREAT, stat.S_IWUSR), "w") as f:
        f.write(_memory(snapshot))
