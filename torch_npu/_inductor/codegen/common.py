import torch._inductor.codegen.common
from torch._inductor.codegen.common import DeviceOpOverrides, device_op_overrides_dict


def get_device_op_overrides(device: str) -> DeviceOpOverrides:
    if not device_op_overrides_dict:
        # remove cuda, xpu device_op_override, add npu device_op_override
        from torch._inductor.codegen import cpu_device_op_overrides, mps_device_op_overrides # noqa: F401
        from .npu import device_op_overrides # noqa: F401

    return device_op_overrides_dict[device]

def patch_get_device_op_overrides():
    torch._inductor.codegen.common.get_device_op_overrides = get_device_op_overrides
    torch._inductor.graph.get_device_op_overrides = get_device_op_overrides
    from torch._inductor.codegen import cpp_wrapper_cpu, cpp_wrapper_gpu
    cpp_wrapper_cpu.get_device_op_overrides = get_device_op_overrides
    cpp_wrapper_gpu.get_device_op_overrides = get_device_op_overrides