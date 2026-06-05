# Owner(s): ["module: tests"]
"""End-to-end: elu/elu_backward are numerically correct through the DVM
backend's native aclnn fallback (elu is excluded from decomposition; the policy
assertions live in test_dvm_decomp.py).

The DVM backend is pinned via torch.compile(options={"npu_backend": "dvm"}).
We deliberately do NOT import torch_npu._inductor at module scope: importing it
loads a backend at import time, which would turn the first torch.compile into a
mid-process backend switch (default -> dvm). Plain ``import torch_npu`` does not
load _inductor.
"""
import unittest

import torch
import torch_npu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not torch.npu.is_available(), "requires an NPU device")
class TestDvmEluNative(TestCase):
    @parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_elu_forward_backward_matches_eager(self, dtype):
        tol = 1e-3 if dtype == torch.float32 else 4e-3
        # build in fp32 then cast (npu normal kernel has no bf16 support)
        ref = torch.randn(256, 4096, device="npu").to(dtype)
        x_e = ref.detach().clone().requires_grad_(True)
        x_c = ref.detach().clone().requires_grad_(True)

        def fn(t):
            return torch.nn.functional.elu(t)

        out_e = fn(x_e)
        out_e.float().sum().backward()

        compiled = torch.compile(
            fn, backend="inductor", options={"npu_backend": "dvm"}
        )
        out_c = compiled(x_c)
        out_c.float().sum().backward()

        self.assertEqual(out_e, out_c, atol=tol, rtol=tol)
        self.assertEqual(x_e.grad, x_c.grad, atol=tol, rtol=tol)


instantiate_parametrized_tests(TestDvmEluNative)
if __name__ == "__main__":
    run_tests()
