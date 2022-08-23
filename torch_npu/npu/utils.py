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

import torch
import torch._six

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


def device_count():
    return torch_npu._C._npu_getDeviceCount()


def set_device(device):
    if isinstance(device, str) and 'npu' in device:
        device = device.replace('npu', torch_npu.npu.native_device)
    if isinstance(device, torch._C.device):
        torch_npu._C._npu_setDevice(device.index)
    elif torch.device(device) :
        torch_npu._C._npu_setDevice(torch.device(device).index)
    else :
        raise AssertionError("input can not convert to torch.device")


def current_device():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDevice()

def get_device_name(device_id: int):
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceProperties(device_id)
    return device_prop.name

def get_device_properties(device_id: int):
    if device_id < 0 or device_id >= device_count():
        raise AssertionError("Invalid device id")
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)

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
            return int(device)
        else:
            device = torch.device(device)
    device_idx = None
    if isinstance(device, torch._C.device):
        # _get_device_index could be called from usrs(device="npu") or inner funcs(device="xla").
        # APIs like torch_npu.npu.synchronize would call torch.device, 
        # which has already changed the key from npu to xla.
        if device.type not in ['npu', torch_npu.npu.native_device]:
            raise ValueError('Expected a npu device, but got: {}'.format(device))
        device_idx = device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch_npu.npu.current_device()
        else:
            raise ValueError('Expected a npu device with a specified index '
                             'or an integer, but got: '.format(device))
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

def get_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_get_float_status(float_status)
    if (float_status.cpu()[0] != 0):
        return True
    else:
        return False

def clear_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_clear_float_status(float_status)
