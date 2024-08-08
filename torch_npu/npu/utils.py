# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import traceback
import contextlib
import threading
from multiprocessing.util import register_after_fork as _register_after_fork
import warnings

import torch
import torch._six

import torch_npu
import torch_npu._C
from torch_npu.utils.device_guard import check_is_valid_ordinal
from torch_npu.utils.error_code import ErrCode, pta_error

_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls = []  # don't invoke these until initialization occurs
_in_bad_fork = False  # this global is also used in torch.manual_seed
_original_pid = False


def is_initialized():
    r"""Returns whether PyTorch's NPU state has been initialized."""
    return _initialized and not _in_bad_fork


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
        if _in_bad_fork:
            from sys import version_info
            if version_info < (3, 4):
                msg = ("To use NPU with multiprocessing, you must use Python "
                       "3.4+ and the 'spawn' start method")
            else:
                msg = ("To use NPU with multiprocessing, you must use the "
                       "'spawn' start method")
            raise RuntimeError(
                "Cannot re-initialize NPU in forked subprocess. " + msg)

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
    global _initialized, _in_bad_fork
    if _initialized and _original_pid != os.getpid():
        _initialized = False
        _in_bad_fork = True
        torch_npu._C._npu_set_run_yet_variable_to_false()


_register_after_fork(_after_fork, _after_fork)


def synchronize(device=None):
    r"""Waits for all kernels in all streams on a NPU device to complete.

    Arguments:
        device (torch.device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~torch_npu.npu.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    torch_npu.npu._lazy_init()
    with torch_npu.npu.device(device):
        return torch_npu._C._npu_synchronize()


def device_count():
    return torch_npu._C._npu_getDeviceCount()


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


def set_device(device):
    check_is_valid_ordinal(device)
    if isinstance(device, str) and device.startswith("npu"):
        device = device.replace('npu', torch_npu.npu.native_device)
    if isinstance(device, (torch_npu._C.device, torch._C.device)):
        torch_npu._C._npu_setDevice(device.index)
    elif isinstance(device, int):
        torch_npu._C._npu_setDevice(device)
    elif torch.device(str(device)):
        device_index = torch.device(str(device)).index
        check_is_valid_ordinal(device_index)
        torch_npu._C._npu_setDevice(device_index)
    else:
        raise AssertionError("input can not convert to torch.device" + pta_error(ErrCode.TYPE))


def current_device():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDevice()


def get_device_name(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceProperties(device_id)
    return device_prop.name


def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)


def mem_get_info(device=None):
    if device is None:
        device = torch_npu.npu.current_device()
    device_id = _get_device_index(device)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceMemories(device_id)
    return device_prop.free_memory, device_prop.total_memory
    

def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not 
    have a corresponding concept and is not supported. By default, it returns None
    """
    warnings.warn("torch_npu.npu.get_device_capability isn't implemented!")
    return None


def utilization(device=None):
    r"""Query the comprehensive utilization rate of device
    """
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceUtilizationRate(device_id)


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch._six.string_classes):
        if "npu" not in device:
            output = None
            try:
                output = int(device)
                print(f"Warning: Expected an NPU device, but got {device}. "
                      f"Input in this format will be deprecated in future version.")
            except ValueError:
                device = torch.device(device)
            if output is not None:
                return output
        else:
            device = torch.device(device)
    device_idx = None
    if isinstance(device, (torch.device, torch._C.device)):
        # _get_device_index could be called from usrs(device="npu") or inner funcs(device="xla").
        # APIs like torch_npu.npu.synchronize would call torch.device, 
        # which has already changed the key from npu to xla.
        if device.type not in ['npu', torch_npu.npu.native_device]:
            raise ValueError('Expected a npu device, but got: {}'.format(device) + pta_error(ErrCode.VALUE))
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch_npu.npu.current_device()
        else:
            raise ValueError('Expected a npu device with a specified index '
                             'or an integer, but got: '.format(device) + pta_error(ErrCode.VALUE))
    return device_idx


def is_available():
    if (not hasattr(torch_npu._C, '_npu_setDevice')):
        return False
    return device_count() > 0


class device(object):
    r"""Context-manager that changes the selected device.

    Arguments:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        if self.idx == -1:
            return
        self.prev_idx = torch_npu._C._npu_getDevice()
        if self.prev_idx != self.idx:
            torch_npu._C._npu_setDevice(self.idx)
        torch_npu.npu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx != self.idx:
            torch_npu._C._npu_setDevice(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Arguments:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_npu else -1
        super(device_of, self).__init__(idx)


@contextlib.contextmanager
def stream(stream):
    r"""Context-manager that selects a given stream.

    All NPU kernels queued within its context will be enqueued on a selected
    stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    .. note:: Streams are per-device. If the selected stream is not on the
        current device, this function will also change the current device to
        match the stream.
    """
    if stream is None:
        yield
        return
    src_prev_stream = current_stream()

    if src_prev_stream.device != stream.device:
        # The given stream is on a different device; have to restore the
        # current_stream on that device on exit as well
        with device(stream.device):
            dst_prev_stream = current_stream()

    torch_npu._C._npu_setStream(stream._cdata)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            torch_npu._C._npu_setStream(dst_prev_stream._cdata)
        torch_npu._C._npu_setStream(src_prev_stream._cdata)


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_npu.npu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_npu.npu._lazy_init()
    return torch_npu.npu.Stream(_cdata=torch_npu._C._npu_getCurrentStream(
        _get_device_index(device, optional=True)))


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_npu.npu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_npu.npu._lazy_init()
    return torch_npu.npu.Stream(_cdata=torch_npu._C._npu_getDefaultStream(
        _get_device_index(device, optional=True)))


def set_sync_debug_mode(debug_mode):
    r"""Sets the debug mode for npu synchronizing operations.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.

    Warning:
        This is an experimental feature, and not all synchronizing operations will trigger warning or error.
    """

    if isinstance(debug_mode, str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`" + pta_error(ErrCode.PARAM)
            )

    torch_npu._C._npu_set_sync_debug_mode(debug_mode)


def get_sync_debug_mode():
    r"""Returns current value of debug mode for npu synchronizing operations."""

    return torch_npu._C._npu_get_sync_debug_mode()


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name) + pta_error(ErrCode.UNAVAIL)
        )

    return type(name, (object,), {"__init__": init_err})


if not hasattr(torch_npu._C, '_NPUStreamBase'):
    # Define dummy base classes
    torch_npu._C.__dict__['_NPUStreamBase'] = _dummy_type('NPUStreamBase')
    torch_npu._C.__dict__['_NPUEventBase'] = _dummy_type('NPUEventBase')


def init_dump():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_initDump()


def set_dump(cfg_file):
    torch_npu.npu._lazy_init()
    cfg_file_path = os.path.realpath(cfg_file)
    return torch_npu._C._npu_setDump(cfg_file_path)


def finalize_dump():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_finalizeDump()


def get_soc_version():
    torch_npu.npu._lazy_init()
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version


def is_support_inf_nan():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_is_support_inf_nan()


def is_bf16_supported():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_is_bf16_supported()


def get_npu_overflow_flag():
    if is_support_inf_nan():
        raise RuntimeError("Unsupport api when soc_version >= Ascend910B1, please use npu_check_overflow" +
                           pta_error(ErrCode.NOT_SUPPORT))
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_get_float_status(float_status)
    if (result.cpu()[0] != 0):
        return True
    else:
        return False


def npu_check_overflow(grad):
    if is_support_inf_nan():
        if isinstance(grad, float):
            cpu_sum = grad
        elif isinstance(grad, torch.Tensor):
            cpu_sum = float(grad.float().sum())
        else:
            raise RuntimeError("Unsupport type." + pta_error(ErrCode.TYPE))

        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        else:
            return False
    else:
        ret = get_npu_overflow_flag()
        if ret:
            clear_npu_overflow_flag()
        return ret


def clear_npu_overflow_flag():
    if is_support_inf_nan():
        warnings.warn("When soc_version >= Ascend910B1, clear_npu_overflow_flag is useless, please remove it.")
        return
    float_status = torch.zeros(8).npu()
    torch_npu.npu_clear_float_status(float_status)


def current_blas_handle():
    warnings.warn("NPU does not use blas handle.")
    return None


def stress_detect():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_stress_detect()
