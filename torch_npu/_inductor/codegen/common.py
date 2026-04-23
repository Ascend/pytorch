import torch._inductor.codegen.common
from torch._inductor.codegen.common import device_op_overrides_dict


def register_device_op_overrides_npu():
    if not device_op_overrides_dict:
        from torch._inductor.codegen import cpu_device_op_overrides, mps_device_op_overrides # noqa: F401
        from torch_npu._inductor import npu_device # noqa: F401
    elif "npu" not in device_op_overrides_dict:
        from torch_npu._inductor import npu_device
