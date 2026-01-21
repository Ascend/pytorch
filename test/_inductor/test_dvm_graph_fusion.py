from unittest import skip
import torch

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import (
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)
from torch_npu._inductor.dvm.graph_fusion import DvmGraphFusionPatch


class TestModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b, c):
        b = torch.transpose(b, 0, 1)
        add = a + b
        mul = add * c
        return mul, torch.sum(mul, dim=[0, 2], keepdim=True)


class MatMulModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, a, b):
        mm = torch.mm(a.t(), b)
        return mm + 3


class TestDvmByGraphFusion(TestCase):
    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_basic_partitioning(self, dtype, is_dynamic):
        a = torch.normal(0, 0.1, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.1, size=(512, 4, 1024), dtype=torch.float16).npu()
        c = torch.normal(0, 0.1, size=(1, 1024), dtype=dtype).npu()
        model = TestModule()

        with DvmGraphFusionPatch():
            dvm_compiled_model = torch.compile(
                model, backend="inductor", dynamic=is_dynamic
            )
            with torch.no_grad():
                expect = model(a, b, c)
                result = dvm_compiled_model(a, b, c)
                self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)

    @parametrize("k,n,m", [(1280, 2560, 640), (1024, 1280, 2560)])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_matmul(self, k, n, m, dtype, is_dynamic):
        a = torch.normal(0, 0.02, size=(k, n), dtype=dtype).npu()
        b = torch.normal(0, 0.02, size=(k, m), dtype=dtype).npu()
        model = MatMulModule()

        with DvmGraphFusionPatch():
            dvm_compiled_model = torch.compile(
                model, backend="inductor", dynamic=is_dynamic
            )
            with torch.no_grad():
                expect = model(a, b)
                result = dvm_compiled_model(a, b)
                self.assertEqual(expect, result, atol=2e-3, rtol=2e-3)


instantiate_parametrized_tests(TestDvmByGraphFusion)

if __name__ == "__main__":
    run_tests()
