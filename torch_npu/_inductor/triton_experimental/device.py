# Copyright (c) 2026, Huawei Technologies Co., Ltd
#
"""NPU device wiring for the triton_experimental backend.

Registers the NPU device with Inductor and Dynamo: the Triton scheduling /
wrapper backend, the device-op overrides (device_guard / raw-stream / ...), and
the Dynamo device interface. Kept out of ``__init__`` so package import stays a
thin activation entry point; ``_activate`` calls the ``register_*`` helpers.
"""
import torch
from torch._inductor.codegen.common import register_device_op_overrides
from torch._dynamo.device_interface import register_interface_for_device
from torch_npu.utils._dynamo_device import NpuInterface


def register_backend_for_npu():
    # Imported here: ``torch_npu._inductor.codegen`` pulls in Inductor internals
    # that would otherwise be resolved while this package is still importing.
    from torch_npu._inductor.compile_options import register_npu_backend

    from . import config as ncfg
    from .codegen import cpp_wrapper as cpp_codegen
    from .codegen import triton as triton_codegen
    from .codegen import wrapper as py_wrapper_codegen
    # torch without pytorch#194998: ncfg keeps the device config slot; with
    # the feature, compile_options owns the slot for "npu.*" option routing
    # and ncfg stays registered under its own module name.
    register_npu_backend(
        triton_codegen.NPUTritonScheduling,
        py_wrapper_codegen.NPUWrapperCodeGen,
        cpp_codegen.TritonCppWrapperNpu,
        backend_config=ncfg,
    )


from torch_npu._inductor.codegen.npu.device_op_overrides import (
    NewNPUDeviceOpOverrides as NPUDeviceOpOverrides,
)

# Resolve the raw-stream C binding once. NoWait variant skips an internal
# stream-sync wait when present; both return the device stream handle directly.
try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _npu_getCurrentRawStream
except ImportError:
    from torch_npu._C import _npu_getCurrentRawStream


def get_current_raw_stream(device):
    # Hot path (once per launch). current_stream() builds a full Stream object
    # (~19us) just to read one int; the raw C binding returns it directly (~0.6us).
    return _npu_getCurrentRawStream(device)


class NewNPUDeviceOpOverrides(NPUDeviceOpOverrides):
    def import_get_raw_stream_as(self, name):
        return f"from torch_npu._inductor.triton_experimental import get_current_raw_stream as {name}"


def register_device_op_overrides_for_npu():
    register_device_op_overrides('npu', NewNPUDeviceOpOverrides())


# Override the original dynamo device interface in torch_npu. NpuInterface already
# provides is_available/get_compute_capability/get_raw_stream; subclass only to
# redirect get_raw_stream to our wrapper.
class NewNpuInterface(NpuInterface):

    @staticmethod
    def is_available() -> bool:
        from torch_npu.npu import device_count
        return device_count() > 0

    @staticmethod
    def get_compute_capability(device=None):
        return torch.npu.get_device_name(device)

    get_raw_stream = staticmethod(get_current_raw_stream)


def register_interface_for_npu():
    register_interface_for_device("npu", NewNpuInterface)
