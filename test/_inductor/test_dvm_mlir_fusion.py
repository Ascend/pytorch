import os
from unittest import skip

import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
import torch_npu


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        b = torch.transpose(b, 0, 1)
        add = a + b
        sub = c - a
        mul = add * sub
        mul = mul + 3
        return mul, torch.sum(mul, dim=[0, 2], keepdim=True)


@skip("request torch-mlir")
class TestDvmByMlir(TestCase):
    os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'mlir'

    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_basic_partitioning(self, dtype, is_dynamic):
        a = torch.normal(0, 0.01, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=dtype).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=dtype).npu()
        model = TestModule()
        from torch_npu._inductor.dvm import mlir_fusion

        dvm_compiled_model = torch.compile(
            model, backend="inductor", dynamic=is_dynamic
        )
        with torch.no_grad():
            expect = model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestDvmByMlir)
if __name__ == "__main__":
    run_tests()
