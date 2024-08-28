# Owner(s): ["module: dynamo"]
import sys

import torch
from torch._dynamo.test_case import TestCase

import torch_npu


class TestNpu(TestCase):
    def test_optimize_npu(self):
        func = torch.ops.aten.relu.default
        dynamo_func = torch._dynamo.optimize('npu')(func)

        x = torch.randn(10).npu()
        eager_result = func(x)
        dynamo_result = dynamo_func(x)
        self.assertEqual(eager_result, dynamo_result)

    def test_npu_inductor(self):
        func = torch.compile(torch.ops.aten.relu.default)

        x = torch.randn(10).npu()
        with self.assertRaisesRegex(RuntimeError, "Device npu not supported"):
            func(x)
 
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
