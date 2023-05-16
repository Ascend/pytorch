__all__ = [
    "is_initialized", "_lazy_call", "_lazy_init", "init", "set_dump",
    "synchronize", "device_count", "set_device", "current_device", "get_device_name",
    "get_device_properties", "get_device_capability", "_get_device_index", "is_available", "device", "device_of",
    "stream", "set_stream", "current_stream", "default_stream", "init_dump",
    "finalize_dump", "manual_seed", "manual_seed_all",
    "seed", "seed_all", "initial_seed", "_free_mutex", "caching_allocator_alloc",
    "caching_allocator_delete", "set_per_process_memory_fraction", "empty_cache", "memory_stats",
    "memory_stats_as_nested_dict", "reset_accumulated_memory_stats",
    "reset_peak_memory_stats", "reset_max_memory_allocated", "reset_max_memory_cached",
    "memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved",
    "memory_cached", "max_memory_cached", "memory_snapshot", "memory_summary",
    "Stream", "Event", "profiler", "set_option", "set_aoe", "profile", "prof_init",
    "prof_start", "prof_stop", "prof_finalize", "iteration_start", "iteration_end",
    "profileConfig", "_is_in_bad_fork", "set_compile_mode",
    "FloatTensor", "IntTensor", "DoubleTensor", "LongTensor", "ShortTensor", 
    "CharTensor", "ByteTensor", "HalfTensor", "set_mm_bmm_format_nd", "get_mm_bmm_format_nd",
    "get_npu_overflow_flag", "clear_npu_overflow_flag", "get_rng_state", "set_rng_state",
    "get_rng_state_all", "set_rng_state_all", "make_replay_graph", "is_jit_compile_false",
    "dump_enable", "dump_disable", "get_amp_supported_dtype", "is_autocast_enabled", "set_autocast_enabled",
    "get_autocast_dtype", "set_autocast_dtype"
]

from typing import Tuple, Union
from multiprocessing.util import register_after_fork as _register_after_fork
import traceback
import threading
import os
import torch
import torch_npu
from .utils import (synchronize, device_count, set_device, current_device, get_device_name,
                    get_device_properties, get_device_capability, _get_device_index,
                    device, device_of, stream, set_stream, current_stream, default_stream, init_dump,
                    finalize_dump, set_dump, get_npu_overflow_flag, clear_npu_overflow_flag)
from .streams import Stream, Event

from . import profiler
from .npu_config import *  # noqa: F403
from .datadump import dump_enable, dump_disable
from .autocast_utils import *  # noqa: F403

torch.optim.Optimizer._hook_for_profile = profiler._hook_for_profile
config = npu_config.npuConfig()

matmul = npu_config.allowHF32Matmul()
conv = npu_config.allowHF32Conv()

default_generators: Tuple[torch._C.Generator] = ()  # type: ignore[assignment]

_is_internal_in_bad_fork = False
_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_original_pid = False

def _is_in_bad_fork():
    return _is_internal_in_bad_fork

def is_initialized():
    r"""Returns whether PyTorch's NPU state has been initialized."""
    return _initialized and not _is_internal_in_bad_fork


def _lazy_call(callable):
    if _initialized:
        callable()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((callable, traceback.format_stack()))


class DeferredNpuCallError(Exception):
    pass


def init():
    r"""Initialize PyTorch's NPU state.  You may need to call
    this explicitly if you are interacting with PyTorch via
    its C API, as Python bindings for NPU functionality will not
    be until this initialization takes place.  Ordinary users
    should not need this, as all of PyTorch's NPU methods
    automatically initialize NPU state on-demand.

    Does nothing if the NPU state is already initialized.
    """
    torch_npu.npu._lazy_init()


def _lazy_init():
    def _queue_call(queued_calls):
        for queued_call, orig_traceback in queued_calls:
            try:
                queued_call()
            except Exception as e:
                msg = (f"NPU call failed lazily at initialization with error: {str(e)}\n\n"
                        f"NPU call was originally invoked at:\n\n{orig_traceback}")
                raise DeferredNpuCallError(msg) from e

    global _initialized, _original_pid, _queued_calls
    if _initialized or hasattr(_tls, 'is_initializing'):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if _initialized:
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_internal_in_bad_fork:
            raise RuntimeError(
                "Cannot re-initialize NPU in forked subprocess. To use NPU with "
                "multiprocessing, you must use the 'spawn' start method")

        torch_npu._C._npu_init()

        _original_pid = os.getpid()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True
        try:
            _queue_call(_queued_calls)
        finally:
            delattr(_tls, 'is_initializing')
        _initialized = True


def _after_fork(arg):
    global _initialized, _is_internal_in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _is_internal_in_bad_fork = True
        torch._C._npu_set_run_yet_variable_to_false()

_register_after_fork(_after_fork, _after_fork)

from .graph import is_graph_mode, disable_graph_mode, enable_graph_mode, launch_graph
from .replay_graph import make_replay_graph

def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device('npu', device)
    return device

def _get_generator(device: torch.device) -> torch._C.Generator:
    r"""Return the NPU Generator object for the given device.

    Args:
        device (torch.device): selected device.
    """

    idx = device.index
    if idx is None:
        idx = current_device()
    return torch.npu.default_generators[idx]

def _set_rng_state_offset(offset: int, device: Union[int, str, torch.device] = 'npu') -> None:
    r"""Sets the random number generator state offset of the specified NPU.

    Args:
        offset (int): The desired offset
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).
    """
    final_device = _get_device(device)

    def cb():
        default_generator = _get_generator(final_device)
        default_generator.set_offset(offset)

    _lazy_call(cb)

def _get_rng_state_offset(device: Union[int, str, torch.device] = 'npu') -> int:
    r"""Returns the random number generator state offset of the specified NPU.

    Args:
        device (torch.device or int, optional): The device to return the RNG state offset of.
            Default: ``'npu'`` (i.e., ``torch.device('npu')``, the current NPU device).

    .. warning::
        This function eagerly initializes NPU.
    """
    _lazy_init()
    final_device = _get_device(device)
    default_generator = _get_generator(final_device)
    return default_generator.get_offset()

def is_available():
    if (not hasattr(torch_npu._C, '_npu_setDevice')):
        return False
    return device_count() > 0

from .random import *  # noqa: F403
from .memory import *  # noqa: F403
