import torch_npu


def patch_device_to_aten():
    from torch._inductor import codegen
    codegen.cpp_utils.DEVICE_TO_ATEN["npu"] = "at::kPrivateUse1"
