import unittest
import torch
import torch_npu

from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from testutils import TestUtils


class TestInductorStaticKernel(TestUtils):

    def simple_op(self, x):
        return torch.neg(x)

    @parametrize("shape", [(1024, 1024), (4096,)])
    @parametrize("dtype", [torch.float16, torch.float32])
    def test_inductor_static_kernel(self, shape, dtype):
        device = "npu"

        x = torch.randn(shape, dtype=dtype, device=device)

        ref = self.simple_op(x)

        torch._inductor.config.triton.cudagraph_trees = False
        torch_npu.npu.aclnn._use_static_aclnn_kernel = True

        compiled_fn = torch.compile(
            self.simple_op,
            backend="inductor",
            dynamic=False
        )

        for _ in range(3):
            compiled_fn(x)

        torch.npu.synchronize()

        out = compiled_fn(x)
        torch.npu.synchronize()

        self.assertEqual(ref, out)

instantiate_parametrized_tests(TestInductorStaticKernel)

if __name__ == "__main__":
    torch.npu.config.allow_internal_format = False
    run_tests()