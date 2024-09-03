__all__ = [
    "is_initialized",
    "init",
    "set_dump",
    "synchronize",
    "device_count",
    "can_device_access_peer",
    "set_device",
    "current_device",
    "get_device_name",
    "get_device_properties",
    "get_device_capability",
    "mem_get_info",
    "is_available",
    "device",
    "device_of",
    "stream",
    "set_stream",
    "current_stream",
    "default_stream",
    "set_sync_debug_mode",
    "get_sync_debug_mode",
    "init_dump",
    "utilization",
    "finalize_dump",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
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
    "Stream",
    "Event",
    "set_option",
    "set_aoe",
    "set_compile_mode",
    "set_mm_bmm_format_nd",
    "get_mm_bmm_format_nd",
    "get_npu_overflow_flag",
    "clear_npu_overflow_flag",
    "stress_detect",
    "get_rng_state",
    "set_rng_state",
    "get_rng_state_all",
    "set_rng_state_all",
    "is_jit_compile_false",
    "get_amp_supported_dtype",
    "is_autocast_enabled",
    "is_bf16_supported",
    "set_autocast_enabled",
    "get_autocast_dtype",
    "set_autocast_dtype",
    "BoolStorage",
    "ByteStorage",
    "ShortStorage",
    "LongStorage",
    "IntStorage",
    "HalfStorage",
    "CharStorage",
    "DoubleStorage",
    "FloatStorage",
    "BoolTensor",
    "ByteTensor",
    "CharTensor",
    "DoubleTensor",
    "FloatTensor",
    "HalfTensor",
    "IntTensor",
    "LongTensor",
    "ShortTensor",
    "BFloat16Tensor",
    "BFloat16Storage",
    "current_blas_handle",
    "stop_device",
    "restart_device",
    "check_uce_in_memory",
    "config",
    "matmul",
    "conv",
    "enable_deterministic_with_backward",
    "disable_deterministic_with_backward",
    "enable_flash_sdp",
    "enable_math_sdp",
    "enable_mem_efficient_sdp",
    "flash_sdp_enabled",
    "math_sdp_enabled",
    "mem_efficient_sdp_enabled",
    "preferred_linalg_library",
    "sdp_kernel",
    "mstx",
    "SyncLaunchStream"
]

from typing import Tuple, Union
from multiprocessing.util import register_after_fork as _register_after_fork
import traceback
import threading
import os
import torch
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from torch._utils import classproperty
from torch_npu.utils import should_print_warning

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error, prof_error
from .utils import (synchronize, device_count, can_device_access_peer, set_device, current_device, get_device_name,
                    get_device_properties, get_device_capability, _get_device_index, 
                    device, device_of, stream, set_stream, current_stream, default_stream, set_sync_debug_mode,
                    get_sync_debug_mode, init_dump, current_blas_handle, is_bf16_supported,
                    utilization, finalize_dump, set_dump, get_npu_overflow_flag, clear_npu_overflow_flag, mem_get_info,
                    check_uce_in_memory, stress_detect)
from ._recovery import restart_device, stop_device
from .streams import Stream, Event, SyncLaunchStream
from .mstx import mstx
from .npu_config import *  # noqa: F403
from .autocast_utils import *  # noqa: F403
from .backends import *  # noqa: F403
from .deterministic import enable_deterministic_with_backward, disable_deterministic_with_backward # noqa: F403

# init profiler
if not torch_npu._C._profiler_init():
    raise RuntimeError("proflier initialization failed" + prof_error(ErrCode.UNAVAIL))

config = npu_config._npuConfig()

matmul = npu_config._allowHF32Matmul()
conv = npu_config._allowHF32Conv()

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


def _lazy_call(cb):
    if _initialized:
        cb()
    else:
        # Don't store the actual traceback to avoid memory cycle
        _queued_calls.append((cb, traceback.format_stack()))


class _DeferredNpuCallError(Exception):
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
                raise DeferredNpuCallError(msg + pta_error(ErrCode.INTERNAL)) from e

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
        torch_npu._C._npu_set_run_yet_variable_to_false()


_register_after_fork(_after_fork, _after_fork)


def _get_device(device: Union[int, str, torch.device]) -> torch.device:
    r"""Return the torch.device type object from the passed in device.

    Args:
        device (torch.device or int): selected device.
    """
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device('npu', device)
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


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _NPUBase.__new__
    return super(_NPUBase, cls).__new__(cls, *args, **kwargs)


class _NPUBase:
    is_npu = True
    is_sparse = False

    def type(self, *args, **kwargs):
        with device(self.get_device()):
            return super().type(*args, **kwargs)

    __new__ = _lazy_new


class _NPULegacyStorage(_LegacyStorage):
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        raise RuntimeError('from_buffer: Not available for NPU storage' + pta_error(ErrCode.UNAVAIL))

    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs):
        raise RuntimeError('_new_with_weak_ptr: Not available for NPU storage' + pta_error(ErrCode.UNAVAIL))

    @classmethod
    def _new_shared_filename(cls, manager, obj, size, *, device=None, dtype=None):
        raise RuntimeError('_new_shared_filename: Not available for NPU storage' + pta_error(ErrCode.UNAVAIL))


class ByteStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bfloat16


del _LegacyStorage
del _NPULegacyStorage

torch._storage_classes.add(DoubleStorage)
torch._storage_classes.add(FloatStorage)
torch._storage_classes.add(LongStorage)
torch._storage_classes.add(IntStorage)
torch._storage_classes.add(ShortStorage)
torch._storage_classes.add(CharStorage)
torch._storage_classes.add(ByteStorage)
torch._storage_classes.add(HalfStorage)
torch._storage_classes.add(BoolStorage)
torch._storage_classes.add(BFloat16Storage)
