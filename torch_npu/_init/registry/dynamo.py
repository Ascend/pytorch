from torch._dynamo.device_interface import register_interface_for_device

from torch_npu.utils._dynamo_device import NpuInterface


def _dynamo_register_interface_for_device():
    register_interface_for_device("npu", NpuInterface)
    for i in range(32):
        register_interface_for_device(f"npu:{i}", NpuInterface)


def register_dynamo_backends():
    from torch_npu.dynamo import _register_backends

    _register_backends()


def register_dynamo_device_interface():
    """
    Register NPU device interface for Dynamo
    """
    _dynamo_register_interface_for_device()


def register_dynamo_trace_rules():
    """
    # Support stream into Dynamo charts. Enable Dynamo to recognize NPU
    stream/device/memory/random APIs and related torch_npu._C bindings during graph capture.
    """
    from torch_npu.dynamo.trace_rule import _patch_npu_trace_rules

    _patch_npu_trace_rules()
