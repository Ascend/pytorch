# Owner(s): ["module: tests"]

import unittest
import os
import torch
from torch.testing._internal.common_utils import run_tests, TestCase, load_tests
from torch._inductor.utils import run_and_get_code
import torch_npu
import torch_npu.testing

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class TestSynchronizeSkip(TestCase):

    def test_synchronize_not_in_compiled_graph(self):

        def func_with_synchronize(x):
            y = x + 1.0
            torch_npu.npu.utils.synchronize()
            return y * 2.0

        x = torch.randn(32, 16, device="npu", dtype=torch.float32)
        expected = (x + 1.0) * 2.0
        compiled_func = torch.compile(func_with_synchronize, backend="inductor", dynamic=False)
        result, inductor_code_list = run_and_get_code(compiled_func, x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
        full_code = "\n".join(inductor_code_list)
        self.assertNotIn("synchronize", full_code)
        self.assertIn("async_compile.triton", full_code)


if __name__ == "__main__":
    run_tests()
