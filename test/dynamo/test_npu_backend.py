# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch._dynamo.test_case import TestCase

import torch_npu


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNpuBackend(TestCase):
    def test_optimize_npu(self):
        func = torch.ops.aten.relu.default
        dynamo_func = torch._dynamo.optimize('npu')(func)

        x = torch.randn(10).npu()
        eager_result = func(x)
        dynamo_result = dynamo_func(x)
        self.assertEqual(eager_result, dynamo_result)

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A', "capture is not supported on 910A, skip this ut.")
    def test_npugraph_ex_backend(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        compiled_model = torch.compile(Model().npu(), backend="npugraph_ex", fullgraph=True, dynamic=False)
        x = torch.ones(1, dtype=torch.int32).npu()
        y = torch.ones(1, dtype=torch.int32).npu()
        z = compiled_model(x, y)
        self.assertEqual(z.item(), 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
