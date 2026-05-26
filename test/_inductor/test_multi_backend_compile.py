# Owner(s): ["module: tests"]
import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import torch_npu


class TestMultiBackendMixedCompile(TestCase):
    @parametrize("dtype", [torch.float32])
    def test_mixed_decorators_in_one_scope(self, dtype):
        @torch.compile(options={"npu_backend": "mlir"})
        def op_sub(a, b):
            return a - b

        @torch.compile()
        def op_add(x, y):
            return x + y

        a = torch.randn(2, 2, dtype=dtype, device="npu")
        b = torch.randn(2, 2, dtype=dtype, device="npu")
        x = torch.randn(3, 4, dtype=dtype, device="npu")
        y = torch.randn(3, 4, dtype=dtype, device="npu")

        sub_out, sub_codes = run_and_get_code(op_sub, a, b)
        self.assertEqual(a - b, sub_out, atol=1e-3, rtol=1e-3)
        self.assertIn("mlir", sub_codes[0])

        add_out, add_codes = run_and_get_code(op_add, x, y)
        self.assertEqual(x + y, add_out, atol=1e-3, rtol=1e-3)
        self.assertIn("triton", add_codes[0])


instantiate_parametrized_tests(TestMultiBackendMixedCompile)

if __name__ == "__main__":
    run_tests()
