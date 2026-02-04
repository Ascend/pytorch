# Owner(s): ["module: dynamo"]
import dataclasses
import functools
import unittest
from typing import List

import torch
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import same
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.pattern_matcher import Match
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_npu

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@unittest.skipIf(DEVICE_NAME == 'Ascend910A', "capture is not supported on 910A, skip this ut.")
class TestNpuGraphEx(TestCase):
    def test_backend(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        compiled_model = torch.compile(Model().npu(), backend="npugraph_ex", fullgraph=True, dynamic=False)
        x = torch.ones(1, dtype=torch.int32, device="npu")
        y = torch.ones(1, dtype=torch.int32, device="npu")
        z = compiled_model(x, y)
        self.assertEqual(z.item(), 2)

    def test_compile_fx(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        def my_backend(gm: torch.fx.GraphModule, example_inputs):
            compiler = torch.npu.npugraph_ex.compile_fx()
            return aot_module_simplified(gm, example_inputs, fw_compiler=compiler)

        compiled_model = torch.compile(Model().npu(), backend=my_backend, fullgraph=True, dynamic=False)
        x = torch.ones(1, dtype=torch.int32, device="npu")
        y = torch.ones(1, dtype=torch.int32, device="npu")
        z = compiled_model(x, y)
        self.assertEqual(z.item(), 2)

    def test_cache_compile(self):
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

    def test_register_replacement(self):
        def search_fn(x1, x2, gamma):
            x_out = torch.add(x1, x2)
            y, _ = torch_npu.npu_rms_norm(x_out, gamma)
            return y, x_out

        def replace_fn(x1, x2, gamma):
            y, _, x_out = torch_npu.npu_add_rms_norm(
                x1, x2, gamma
            )
            return y, x_out

        def extra_check(match: Match):
            x1 = match.kwargs.get("x1")

            if x1 is None:
                return False
            if not hasattr(x1, "meta") or "val" not in x1.meta:
                return False

            a_shape = x1.meta["val"].shape
            return a_shape[-1] == 7168

        fake_mode = FakeTensorMode()
        with fake_mode:
            input_tensor = functools.partial(torch.empty, (1, 1, 2), dtype=torch.float16, device="npu")
            kwargs_tensor = functools.partial(torch.empty, 2, dtype=torch.float16, device="npu")

            torch.npu.npugraph_ex.register_replacement(
                search_fn=search_fn,
                replace_fn=replace_fn,
                example_inputs=(input_tensor(), input_tensor(), kwargs_tensor()),
                extra_check=extra_check
            )

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, data1, data2, gamma):
                x_out = torch.add(data1, data2)
                y, _ = torch_npu.npu_rms_norm(x_out, gamma)

                abs_01 = torch.abs(y)
                sqrt_01 = torch.sqrt(x_out)
                return abs_01, sqrt_01

        model = Model().npu()

        x1 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
        x2 = torch.randn(1, 1, 7168, dtype=torch.float16, device='npu')
        gamma = torch.ones(7168, dtype=torch.float16, device='npu')

        model_compile = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
        res = model_compile(x1, x2, gamma)
        self.assertEqual(res[0].shape[2], 7168)
        self.assertEqual(res[1].shape[2], 7168)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()