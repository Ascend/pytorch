import torch._inductor.codegen.common
from torch._inductor.codegen.common import device_op_overrides_dict


def register_device_op_overrides_npu():
    if not device_op_overrides_dict:
        # remove cuda, xpu device_op_override, add npu device_op_override
        from torch._inductor.codegen import cpu_device_op_overrides, mps_device_op_overrides # noqa: F401
        from .npu import device_op_overrides # noqa: F401
    elif "npu" not in device_op_overrides_dict:
        from .npu import device_op_overrides
