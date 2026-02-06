from unittest import skip
import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

from torch_npu._inductor import dvm


def fused_add_sum(k: dvm.Kernel):
    x = k.load([-1, -1, -1], dvm.float32)
    y = k.load([-1, -1, -1], dvm.float32)
    scalar = k.scalar(dvm.float32)
    a = k.add(x, y)
    b = k.add(a, scalar)
    c = k.sum(b, [0, 1], True)
    k.store(c)


class TestDvmKernelOp(TestCase):
    def test_dvm_kernel_op(self):
        a = torch.normal(0, 0.1, size=(512, 128, 256), dtype=torch.float32).npu()
        b = torch.normal(0, 0.1, size=(512, 1, 256), dtype=torch.float32).npu()
        scalar = 1.22
        expect = torch.sum((a + b + 1.22), dim=[0, 1], keepdim=True)
        result = torch.empty((1, 1, 256), device="npu")
        kernel1 = dvm.kernel(ktype="vector", dyn_shape=True)(fused_add_sum)
        kernel1.kobj.set_kernel_info("dvm_add_sum", "dvm_add_sum", [True] * 4)
        kernel1.run(a, b, scalar, result)
        self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
        kernel2 = dvm.kernel(ktype="split", dyn_shape=True)(fused_add_sum)
        result = kernel2(a, b, scalar)
        self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
