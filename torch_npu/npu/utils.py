import os
from typing import Any, Optional
import warnings
from enum import Enum

import torch
from torch._utils import _get_device_index as _torch_get_device_index

import torch_npu
import torch_npu._C
from torch_npu.utils._error_code import ErrCode, pta_error, _except_handler
from torch_npu.npu._backends import get_soc_version


__all__ = ["synchronize", "device_count", "can_device_access_peer", "set_device", "current_device", "get_device_name",
           "get_device_properties", "mem_get_info", "get_device_capability", "utilization", "device", "device_of", "StreamContext",
           "stream", "set_stream", "current_stream", "default_stream", "set_sync_debug_mode", "get_sync_debug_mode",
           "init_dump", "set_dump", "finalize_dump", "is_support_inf_nan", "is_bf16_supported",
           "get_npu_overflow_flag", "npu_check_overflow", "clear_npu_overflow_flag", "current_blas_handle",
           "check_uce_in_memory", "stress_detect", "get_cann_version"]


def get_cann_version(module="CANN"):
    r"""
    Args:
        module: can be selected from [\"CANN\", \"RUNTIME\", \"COMPILER\", \"HCCL\", \"TOOLKIT\", \"OPP\", \"OPP_KERNEL\", \"DRIVER\"]

    Returns: current version.

    """
    return torch_npu._C._get_cann_version(module)


def _is_gte_cann_version(version, module="CANN"):
    r"""
    compare current cann_version and version.
    Args:
        version: the features are supported or not from which cann version.
        module: can be selected from [\"CANN\", \"RUNTIME\", \"COMPILER\", \"HCCL\", \"TOOLKIT\", \"OPP\", \"OPP_KERNEL\"]

    Returns: If current_version >= version, return True, else return False.

    """
    result = torch_npu._C._is_gte_cann_version(version, module)
    return True if result else False


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


def device_count() -> int:
    return torch_npu.npu.device_count()


def can_device_access_peer(device_id, peer_device_id):
    r"""Checks if peer access between two devices is possible.
    """
    device_id = _get_device_index(device_id, optional=True)
    peer_device_id = _get_device_index(peer_device_id, optional=True)
    if device_id < 0 or device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid devide id" + pta_error(ErrCode.VALUE))
    if peer_device_id < 0 or peer_device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid peer devide id" + pta_error(ErrCode.VALUE))
    return torch_npu._C._npu_canDeviceAccessPeer(device_id, peer_device_id)


def set_device(device):
    device_id = _get_device_index(device, optional=True)
    if device_id >= 0:
        torch_npu._C._npu_setDevice(device_id)


def current_device():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDevice()


def get_device_name(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    return torch_npu._C._npu_getDeviceName()


def get_device_properties(device_name=None):
    device_id = _get_device_index(device_name, optional=True)
    if device_id < 0 or device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceProperties(device_id)


def mem_get_info(device=None):
    if device is None:
        device = torch_npu.npu.current_device()
    device_id = _get_device_index(device)
    if device_id < 0 or device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    device_prop = torch_npu._C._npu_getDeviceMemories(device_id)
    return device_prop.free_memory, device_prop.total_memory


def get_device_capability(device=None):
    r"""Query the minor and major data of device. Cann does not
    have a corresponding concept and is not supported. By default, it returns None
    """
    warnings.warn("torch.npu.get_device_capability isn't implemented!")
    return None


def utilization(device=None):
    r"""Query the comprehensive utilization rate of device
    """
    device_id = _get_device_index(device, optional=True)
    if device_id < 0 or device_id >= torch_npu.npu.device_count():
        raise AssertionError("Invalid device id" + pta_error(ErrCode.VALUE))
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_getDeviceUtilizationRate(device_id)


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
        self.prev_idx = torch_npu._C._npu_getDeviceWithoutSet()
        if self.prev_idx != self.idx:
            torch_npu._C._npu_setDevice(self.idx)
        torch_npu.npu._lazy_init()

    def __exit__(self, *args):
        if self.prev_idx == -1:
            self.prev_idx = 0
        self.idx = torch_npu._C._npu_maybeExchangeDevice(self.prev_idx)
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
                raise ValueError('Expected a npu or cpu device, but got: {}'.format(device) + pta_error(ErrCode.VALUE))
        elif device.type != 'npu':
            raise ValueError('Expected a npu device, but got: {}'.format(device) + pta_error(ErrCode.VALUE))
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


class StreamContext:
    r"""Context-manager that selects a given stream.

    All NPU kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """
    cur_stream: Optional["torch_npu.npu.Stream"]

    def __init__(self, stream_ctx: Optional["torch_npu.npu.Stream"]):
        self.stream = stream_ctx
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.npu.default_stream()
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.npu.default_stream()
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or NPU device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.npu.current_stream()

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.npu.current_stream(cur_stream.device)
        torch.npu.set_stream(cur_stream)

    def __exit__(self, exec_type: Any, exec_value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no NPU device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.npu.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.npu.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream):
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.

    ..Note:: In eager mode stream is of type Stream class while in JIT it is
      an object of the custom class ``torch.classes.npu.Stream``.
    """
    return StreamContext(stream)


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


def is_support_inf_nan():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_is_support_inf_nan()


def is_bf16_supported():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_is_bf16_supported()


def get_npu_overflow_flag():
    if is_support_inf_nan():
        raise RuntimeError("Unsupported api when soc_version >= Ascend910B1, please use npu_check_overflow" +
                           pta_error(ErrCode.NOT_SUPPORT))
    float_status = torch.zeros(8).npu()
    result = torch_npu.npu_get_float_status(float_status)
    if result.cpu()[0] != 0:
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
            raise RuntimeError("Unsupported type." + pta_error(ErrCode.TYPE))

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


def stress_detect():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_stress_detect()


def current_blas_handle():
    warnings.warn("NPU does not use blas handle.")
    return None


def check_uce_in_memory(device_id):
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_check_uce_in_memory(device_id)


def _get_uce_addr():
    torch_npu.npu._lazy_init()
    return torch_npu._C._npu_get_uce_addr()


def _erase_stream(tensor, stream):
    r"""Remove the tags of the tensor that are used by this stream through the record_stream function.

    The memory can be reused between multiple streams. By default, the record_stream is used to mark the memory pool
    to prevent the reused memory from being returned to the memory pool in advance. Each time the memory pool
    applies for memory, it queries the event on the device to determine whether the operator has been executed and
    can be safely released. However, the combination of host and device has a side effect. When the host is dispatched
    much faster than the device, the peak memory usage may be increased because the device is not completely executed
    when the host is querying.

    This api provides the erase_stream capability with memory pool. The memory can be returned in advance by actively
    erasing and freeing the memory after the event wait. The subsequent operators must be executed after the event wait.
    Therefore, the memory that is released back to the memory pool in advance will not be trampled by the subsequent operators.

    Args:
        tensor(Tensor): The tensor whose tag needs to be removed.
        stream(Stream): The tensor is marked in the stream and the tag needs to be removed in the current operation.

    Warning:
        When the current api is in use, it must be used in conjunction with the event wait method.
        Otherwise, memory trampling behavior may occur.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor should be torch.Tensor, could not be {type(tensor)}" + pta_error(ErrCode.TYPE))
    if not isinstance(stream, torch_npu.npu.Stream):
        raise TypeError(f"stream should be torch_npu.npu.Stream, could not be {type(stream)}" + pta_error(ErrCode.TYPE))
    torch_npu._C._npu_eraseStream(tensor=tensor,
                                stream_id=stream.stream_id,
                                device_index=stream.device_index,
                                device_type=stream.device_type)
