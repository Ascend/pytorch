import unittest
import torch

from testutils import TestUtils
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch._inductor import config
from torch._inductor.utils import run_and_get_code

import torch_npu
import torch_npu._inductor

DEVICE = "npu"


class TestCodegenUpcastToFP32(TestUtils):
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("upcast_flag", [True, False])
    def test_codegen_upcast_to_fp32_emits_cast(self, dtype, upcast_flag):
        @torch.compile(backend="inductor")
        def func(x):
            return torch.abs(x)

        x = torch.randn((1024, 1024), device=DEVICE, dtype=dtype)

        with config.patch("triton.codegen_upcast_to_fp32", upcast_flag):
            opt_func = torch._dynamo.optimize("inductor")(func)
            out, code = run_and_get_code(opt_func, x)

        self.assertTrue(".to(tl.float32)" in code[0])
        self.assertEqual(func(x), opt_func(x))

instantiate_parametrized_tests(TestCodegenUpcastToFP32)

if __name__ == "__main__":
    run_tests()