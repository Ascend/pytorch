# Owner(s): ["module: dynamo"]
import unittest
import dataclasses
from typing import List

import torch
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import same

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

    @unittest.skipIf(DEVICE_NAME == 'Ascend910A', "capture is not supported on 910A, skip this ut.")
    def test_npugraph_ex_cache_compile(self):
        @dataclasses.dataclass
        class InputMeta:
            data: torch.Tensor
            is_prompt: bool

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 1)
                self.linear2 = torch.nn.Linear(2, 1)
                for param in self.parameters():
                    torch.nn.init.ones_(param)
                self.cached_prompt = torch.npu.npugraph_ex.inference.cache_compile(self.prompt)
                self.cached_decode = torch.npu.npugraph_ex.inference.cache_compile(self.decode)

            def forward(self, x: InputMeta, kv: List[torch.Tensor]):
                if x.is_prompt:
                    return self.cached_prompt(x, kv)
                return self.cached_decode(x, kv)

            def _forward(self, x, kv):
                return self.linear2(x.data) + self.linear2(kv[0])

            def prompt(self, x, y):
                return self._forward(x, y)

            def decode(self, x, y):
                return self._forward(x, y)

        x = InputMeta(data=torch.ones(2, 2).npu(), is_prompt=True)
        kv = [torch.ones(2, 2).npu()]
        model = Model().npu()
        res_prompt = model(x, kv)
        x.is_prompt = False
        res_decode = model(x, kv)
        res = torch.empty(2, 1).npu().fill_(6.0)
        self.assertTrue(same(res, res_prompt))
        self.assertTrue(same(res, res_decode))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
