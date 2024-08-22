import torch
import torch_npu


def error_aclop():
    torch.npu.set_compile_mode(jit_compile=True)
    x1 = torch.randn(3).float().npu()
    x2 = torch.randn(1).long().npu()
    x3 = torch.randn(1).float().npu()
    y = torch.addcmul(x1, x2, x3)


error_aclop()
