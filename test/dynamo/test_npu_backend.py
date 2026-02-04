# Owner(s): ["module: dynamo"]
import torch
from torch._dynamo.test_case import TestCase

import torch_npu


class TestNpuBackend(TestCase):
    def test_optimize_npu(self):
        func = torch.ops.aten.relu.default
        dynamo_func = torch._dynamo.optimize('npu')(func)

        x = torch.randn(10).npu()
        eager_result = func(x)
        dynamo_result = dynamo_func(x)
        self.assertEqual(eager_result, dynamo_result)
        

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
