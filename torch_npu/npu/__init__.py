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
    "StreamContext",
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
    "empty_virt_addr_cache",
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
    "MemPool",
    "MemPoolContext",
    "use_mem_pool",
    "get_allocator_backend",
    "NPUPluggableAllocator",
    "change_current_allocator",
    "Stream",
    "Event",
    "set_option",
    "CubeMathType",
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
    "mstx",
    "SyncLaunchStream",
    "NPUGraph",
    "graph",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "make_graphed_callables",
    "ExternalEvent",
    "graph_task_group_begin",
    "graph_task_group_end",
    "graph_task_update_begin",
    "graph_task_update_end",
    "set_device_limit",
    "get_device_limit",
    "set_stream_limit",
    "reset_stream_limit",
    "get_stream_limit",
    "ipc_collect",
    "obfuscation_initialize",
    "obfuscation_finalize",
    "obfuscation_calculate",
    "set_op_timeout_ms",
    "host_empty_cache",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "set_deterministic_level",
    "use_compatible_impl",
    "are_compatible_impl_enabled"
]

from typing import Tuple, Union, List, cast, Optional
from multiprocessing.util import register_after_fork as _register_after_fork
import traceback
import threading
import os
import re
import torch
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from torch._utils import classproperty
from torch_npu.utils import _should_print_warning

import torch_npu
from torch_npu.utils._error_code import ErrCode, pta_error, prof_error
from .utils import (obfuscation_initialize, obfuscation_calculate, obfuscation_finalize, 
                    synchronize, set_device, current_device, _get_device_index,
                    device, device_of, StreamContext, stream, set_stream, current_stream, default_stream, set_sync_debug_mode,
                    get_sync_debug_mode, init_dump, current_blas_handle, is_bf16_supported,
                    finalize_dump, set_dump, get_npu_overflow_flag, clear_npu_overflow_flag,
                    check_uce_in_memory, stress_detect, _get_uce_addr, ipc_collect, set_op_timeout_ms)
from ._recovery import restart_device, stop_device
from .streams import Stream, Event, SyncLaunchStream, ExternalEvent
from .mstx import mstx
from .npu_config import *  # noqa: F403
from .autocast_utils import *  # noqa: F403
from .backends import *  # noqa: F403
from ._backends import *  # noqa: F403
from .deterministic import enable_deterministic_with_backward, disable_deterministic_with_backward # noqa: F403
from . import npugraph_ex

from .graphs import (
    NPUGraph,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
    graph_task_group_begin,
    graph_task_group_end,
    graph_task_update_begin,
    graph_task_update_end,
)

# init profiler
if not torch_npu._C._profiler_init():
    raise RuntimeError("proflier initialization failed" + prof_error(ErrCode.UNAVAIL))

config = npu_config._npuConfig()

matmul = npu_config._allowHF32Matmul()
conv = npu_config._allowHF32Conv()
CubeMathType = npu_config._CubeMathType

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


def _parse_visible_devices() -> Union[List[int], List[str]]:
    r"""Parse ASCEND_RT_VISIBLE_DEVICES environment variable."""
    var = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
    if var is None:
        return list(range(64))

    rc: List[int] = []

    if not var:
        return rc

    # Multiple Device IDs are separated by ',' and cannot contain any other characters.
    # If any other characters are included, only the Device IDs before them will be read
    for idx, c in enumerate(var):
        if not (c.isdigit() or c == ","):
            break
        if idx + 1 == len(var):
            idx += 1

    for elem in var[:idx].split(","):
        if not elem:
            return rc
        x = int(elem)
        rc.append(x)

    return rc


def _raw_device_count_ascend_hal() -> int:
    r"""Return number of devices as reported by ascend_hal or negative value if ascend_hal discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    ascend_hal_h = CDLL("libascend_hal.so")

    dev_count = c_int(-1)
    rc = ascend_hal_h.drvGetDevNum(byref(dev_count))


    if rc != 0:
        warnings.warn("Can't get ascend_hal device count")
        return -1
    del ascend_hal_h
    return dev_count.value


def _device_count_ascend_hal() -> int:
    r"""Return number of devices as reported by ascend_hal taking ASCEND_RT_VISIBLE_DEVICES into account.

    Negative value is returned if ascend_hal discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        raw_cnt = _raw_device_count_ascend_hal()
        if raw_cnt <= 0:
            return raw_cnt
        # Trim the list up to a maximum available device
        for idx, val in enumerate(visible_devices):
            # `rts` need ascending order
            if idx > 0 and val <= visible_devices[idx - 1]:
                return 0
            if cast(int, val) >= raw_cnt:
                return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


_cached_device_count: Optional[int] = None


def device_count() -> int:
    r"""Return the number of NPUs available."""
    global _cached_device_count
    if _cached_device_count is not None:
        return _cached_device_count
    ascend_hal_count = _device_count_ascend_hal()
    r = torch_npu._C._npu_getDeviceCount() if ascend_hal_count < 0 else ascend_hal_count
    # NB: Do not cache the device count prior to NPU initialization, because
    # the number of devices can change due to changes to ASCEND_RT_VISIBLE_DEVICES
    # setting prior to NPU initialization.
    if _initialized:
        _cached_device_count = r
    return r


def _launch_host_func(op_stream, fn, user_data):
    torch_npu._C._launch_host_func(op_stream, fn, user_data)


def _subscribe_report(op_stream):
    torch_npu._C._subscribe_report(op_stream)


def _unsubscribe_report(op_stream):
    torch_npu._C._unsubscribe_report(op_stream)


def can_device_access_peer(device_id, peer_device_id):
    r"""Checks if peer access between two devices is possible.
    """
    device_id = _get_device_index(device_id, optional=True)
    peer_device_id = _get_device_index(peer_device_id, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid devide id" + pta_error(ErrCode.VALUE))
    if peer_device_id < 0 or peer_device_id >= device_count():
        raise AssertionError("Invalid peer devide id" + pta_error(ErrCode.VALUE))
    return torch_npu._C._npu_canDeviceAccessPeer(device_id, peer_device_id)


def get_device_name(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    return torch_npu._C._npu_getDeviceName()


def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)


def mem_get_info(device=None):
    if device is None:
        device = torch_npu.npu.current_device()
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceMemories(device_id)
    return device_prop.free_memory, device_prop.total_memory


_cached_device_capability = None
_cached_device_capability_env = None


def get_device_capability(device=None):
    r"""Query the minor and major data of device.

    This function can be configured via the TORCH_NPU_DEVICE_CAPABILITY environment variable. 
    The format should be "major.minor", e.g., "9.0" or "8.0".

    .. note::
        The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device.

    Args:
        device (torch.device or int, optional): The device parameter has no practical meaning.

    Returns:
        tuple(int, int): the device capability of the device. Returns the tuple(major, minor) configured via 
        TORCH_NPU_DEVICE_CAPABILITY, or None if TORCH_NPU_DEVICE_CAPABILITY not configured.

    Example:
        >>> print(torch_npu.npu.get_device_capability())
        None
        >>> os.environ['TORCH_NPU_DEVICE_CAPABILITY'] = '8.0'
        >>> print(torch_npu.npu.get_device_capability())
        (8, 0)
        >>> print(torch_npu.npu.get_device_capability(0))
        (8, 0)
    """
    global _cached_device_capability, _cached_device_capability_env

    capability_env = os.getenv("TORCH_NPU_DEVICE_CAPABILITY")
    warning_str = "The return value of get_device_capability is only for compatibility with PyTorch and does not represent the actual capability of the NPU device."
    if not capability_env:
        warnings.warn(f"You can set the device capability via the environment variable TORCH_NPU_DEVICE_CAPABILITY. {warning_str}")
        return None

    # Fast path: reuse cached result if the environment variable has not changed
    if _cached_device_capability_env == capability_env and _cached_device_capability is not None:
        return _cached_device_capability

    # Validate the format of the environment variable, expected format is 'major.minor' where major and minor are non-negative integers (e.g., '8.0')
    pattern = r'^(\d+)\.(\d+)$'
    match = re.match(pattern, capability_env)
    if not match:
        warnings.warn(
            f"torch.npu.get_device_capability can't parse TORCH_NPU_DEVICE_CAPABILITY environment variable: {capability_env}. "
            "Expected format is 'major.minor' where major and minor are non-negative integers (e.g., '8.0')."
        )
        return None

    warnings.warn(warning_str)
    major_str, minor_str = match.groups()
    major = int(major_str)
    minor = int(minor_str)
    _cached_device_capability = (major, minor)
    _cached_device_capability_env = capability_env
    return _cached_device_capability


def utilization(device=None):
    r"""Query the comprehensive utilization rate of device
    """
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceUtilizationRate(device_id)


def _aclnn_reselect_static_kernel():
    torch_npu.npu._lazy_init()
    torch_npu._C._aclnn_reselect_static_kernel()


from .random import *  # noqa: F403
from .memory import *  # noqa: F403


@staticmethod
def _lazy_new(cls, *args, **kwargs):
    _lazy_init()
    # We may need to call lazy init again if we are a forked child
    # del _NPUBase.__new__
    return super(_NPUBase, cls).__new__(cls, *args, **kwargs)


def _comm_switch_nic(ranks, useBackup):
    torch_npu.npu.synchronize()
    return torch_npu.distributed.distributed_c10d._comm_switch_nic(ranks, useBackup)


def set_deterministic_level(level):
    warnings.warn("After using 'torch_npu.npu.set_deterministic_level', "
                  "please do not use 'torch.use_deterministic_algorithms' anymore, "
                  "as it may cause unknown errors.")
    if level == 0 and torch.are_deterministic_algorithms_enabled():
        warnings.warn("The current configuration value of 'torch_npu.npu.set_deterministic_level' "
                      "conflicts with 'torch.use_deterministic_algorithms'. "
                      "'torch.use_deterministic_algorithms' has been configured to 'False'")
        torch.use_deterministic_algorithms(False)
    elif level >= 1 and not torch.are_deterministic_algorithms_enabled():
        warnings.warn("The current configuration value of 'torch_npu.npu.set_deterministic_level' "
                      "conflicts with 'torch.use_deterministic_algorithms'. "
                      "'torch.use_deterministic_algorithms' has been configured to 'True'")
        torch.use_deterministic_algorithms(True)
    torch_npu._C._npu_set_deterministic_level(level)


def _get_deterministic_level():
    level = torch_npu._C._npu_get_deterministic_level()
    if level == 0 and torch.are_deterministic_algorithms_enabled():
        level = 1
        torch_npu.npu.set_deterministic_level(level)
        return level
    if level >= 1 and not torch.are_deterministic_algorithms_enabled():
        level = 0     
        torch_npu.npu.set_deterministic_level(level)
        return level
    return level


def use_compatible_impl(is_enable):
    option = {"COMPATIBLE_IMPL": "enable" if is_enable else "disable"}
    torch_npu._C._npu_setOption(option)


def are_compatible_impl_enabled():
    compatible_value = torch_npu._C._npu_getOption("COMPATIBLE_IMPL")
    return compatible_value is not None and compatible_value.decode() == "enable"


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
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.uint8


class DoubleStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.double


class FloatStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.float


class HalfStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.half


class LongStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.long


class IntStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int


class ShortStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.short


class CharStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.int8


class BoolStorage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
            _warn_typed_storage_removal()
        return self._dtype

    @classproperty
    def _dtype(self):
        return torch.bool


class BFloat16Storage(_NPULegacyStorage):
    @classproperty
    def dtype(self):
        if _should_print_warning():
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
