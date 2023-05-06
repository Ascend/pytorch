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
from typing import Any
from functools import lru_cache
from multiprocessing.util import register_after_fork as _register_after_fork
import warnings

import torch
from torch._utils import _get_device_index as _torch_get_device_index

import torch_npu
import torch_npu._C

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
        torch._C._npu_set_run_yet_variable_to_false()

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

@lru_cache(maxsize=1)
def device_count():
    return torch_npu._C._npu_getDeviceCount()


def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >=0:
        torch_npu._C._npu_setDevice(device_id)


def current_device():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDevice()

def get_device_name(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceProperties(device_id)
    return device_prop.name

def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)


def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not 
    have a corresponding concept and is not supported. By default, it returns None
    """
    warnings.warn("torch.npu.get_device_capability isn't implemented!")
    return None

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


def _get_device_index(device: Any, optional: bool = False,
                      allow_cpu: bool = False) -> int:
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a NPU device. Note that for a NPU device without a specified index,
    i.e., ``torch.device('npu')``, this will return the current default NPU
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default NPU
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if allow_cpu:
            if device.type not in ['npu', 'cpu']:
                raise ValueError('Expected a npu or cpu device, but got: {}'.format(device))
        elif device.type != 'npu':
            raise ValueError('Expected a npu device, but got: {}'.format(device))
    if not torch.jit.is_scripting():
        if isinstance(device, torch.npu.device):
            return device.idx
    return _torch_get_device_index(device, optional, allow_cpu)


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

    torch.npu.set_stream(stream)
    try:
        yield
    finally:
        if src_prev_stream.device != stream.device:
            torch.npu.set_stream(dst_prev_stream)
        torch.npu.set_stream(src_prev_stream)


def set_stream(stream):
    r"""Sets the current stream.This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.
    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch_npu._C._npu_setStream(stream_id=stream.stream_id,
                                device_index=stream.device_index,
                                device_type=stream.device_type)


def current_stream(device=None):
    r"""Returns the currently selected :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch_npu.npu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_npu.npu._lazy_init()
    streamdata = torch_npu._C._npu_getCurrentStream(
        _get_device_index(device, optional=True))
    return torch_npu.npu.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def default_stream(device=None):
    r"""Returns the default :class:`Stream` for a given device.

    Arguments:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch_npu.npu.current_device`, if :attr:`device` is ``None``
            (default).
    """
    torch_npu.npu._lazy_init()
    streamdata = torch_npu._C._npu_getDefaultStream(
        _get_device_index(device, optional=True))
    return torch_npu.npu.Stream(stream_id=streamdata[0], device_index=streamdata[1], device_type=streamdata[2])


def _dummy_type(name):
    def init_err(self):
        class_name = self.__class__.__name__
        raise RuntimeError(
            "Tried to instantiate dummy base class {}".format(class_name))
    return type(name, (object,), {"__init__": init_err})


if not hasattr(torch_npu._C, '_NPUStreamBase'):
    # Define dummy base classes
    torch_npu._C.__dict__['_NPUStreamBase'] = _dummy_type('NPUStreamBase')
    torch_npu._C.__dict__['_NPUEventBase'] = _dummy_type('NPUEventBase')

if not hasattr(torch_npu._C, '_NPUReplayGraphBase'):
    # Define dummy base classes
    torch_npu._C.__dict__['_NPUReplayGraphBase'] = _dummy_type('NPUReplayGraphBase')


def init_dump():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_initDump()

def set_dump(cfg_file):
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_setDump(cfg_file)

def finalize_dump():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_finalizeDump()

def get_soc_version():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version

def get_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_get_float_status(float_status)
    if (result.cpu()[0] != 0):
        return True
    else:
        return False

def npu_check_over_flow(grad):
    soc_version = get_soc_version()

    if (soc_version >= 220):
        cpu_sum = float(grad.float().sum())
        if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
            return True
        else:
            return False
    else:
        return get_npu_overflow_flag()

def clear_npu_overflow_flag():
    soc_version = get_soc_version()
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_clear_float_status(float_status)
