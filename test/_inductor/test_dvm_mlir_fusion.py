# Owner(s): ["module: tests"]
import os

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


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


class ReduceCaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, arg0_1, arg1_1, arg2_1):
        sum_1 = torch.ops.aten.sum.dim_IntList(arg0_1, [0, 2, 3], True)
        div = torch.ops.aten.div.Scalar(sum_1, 9800.0)
        view = torch.ops.aten.reshape.default(div, [64])
        mul = torch.ops.aten.mul.Scalar(view, 0.1)
        mul_1 = torch.ops.aten.mul.Scalar(arg1_1, 0.9)
        add = torch.ops.aten.add.Tensor(mul, mul_1)
        expand = torch.ops.aten.expand.default(div, [8, 64, 35, 35])
        sub = torch.ops.aten.sub.Tensor(arg0_1, expand)
        pow_1 = torch.ops.aten.pow.Tensor_Scalar(sub, 2)
        sum_2 = torch.ops.aten.sum.dim_IntList(pow_1, [0, 2, 3], True)
        div_1 = torch.ops.aten.div.Scalar(sum_2, 9800.0)
        add_1 = torch.ops.aten.add.Scalar(div_1, 0.001)
        rsqrt = torch.ops.aten.rsqrt.default(add_1)
        view_1 = torch.ops.aten.reshape.default(div_1, [64])
        mul_2 = torch.ops.aten.mul.Scalar(view_1, 1.0001020512297174)
        mul_3 = torch.ops.aten.mul.Scalar(mul_2, 0.1)
        mul_4 = torch.ops.aten.mul.Scalar(arg2_1, 0.9)
        add_2 = torch.ops.aten.add.Tensor(mul_3, mul_4)
        return (div, add, rsqrt, add_2)


class TestDvmByMlir(TestCase):
    @parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    @parametrize("is_dynamic", [True, False])
    def test_basic_partitioning(self, dtype, is_dynamic):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
        a = torch.normal(0, 0.01, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=dtype).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=dtype).npu()
        model = TestModule()
        dvm_compiled_model = torch.compile(
            model, backend="inductor", dynamic=is_dynamic
        )
        with torch.no_grad():
            expect = model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
        del os.environ["TORCHINDUCTOR_NPU_BACKEND"]

    @parametrize("dtype", [torch.bfloat16])
    @parametrize("is_dynamic", [False])
    def test_basic_partitioning_npugraph(self, dtype, is_dynamic):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
        a = torch.normal(0, 0.01, size=(512, 1), dtype=dtype).npu()
        b = torch.normal(0, 0.01, size=(512, 4, 256), dtype=dtype).npu()
        c = torch.normal(0, 0.01, size=(1, 256), dtype=dtype).npu()
        model = TestModule()
        dvm_compiled_model = torch.compile(
            model,
            backend="inductor",
            dynamic=is_dynamic,
            options={"triton.cudagraphs": True},
        )
        with torch.no_grad():
            expect = model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            result = dvm_compiled_model(a, b, c)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
        del os.environ["TORCHINDUCTOR_NPU_BACKEND"]

    @parametrize("dtype", [torch.float16, torch.float32])
    @parametrize("is_dynamic", [True, False])
    def test_reduce_case(self, dtype, is_dynamic):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
        arg0 = torch.empty_strided(
            torch.Size((8, 64, 35, 35)),
            (78400, 1225, 35, 1),
            dtype=dtype,
            device="npu",
        ).uniform_(0, 1)
        arg1 = torch.empty_strided(
            torch.Size((64,)), (1,), dtype=dtype, device="npu"
        ).uniform_(0, 1)
        arg2 = torch.empty_strided(
            torch.Size((64,)), (1,), dtype=dtype, device="npu"
        ).uniform_(0, 1)
        model = ReduceCaseModel()
        dvm_compiled_model = torch.compile(
            model, backend="inductor", dynamic=is_dynamic
        )
        with torch.no_grad():
            expect = model(arg0, arg1, arg2)
            result = dvm_compiled_model(arg0, arg1, arg2)
            self.assertEqual(expect, result, atol=1e-3, rtol=1e-3)
        del os.environ["TORCHINDUCTOR_NPU_BACKEND"]


instantiate_parametrized_tests(TestDvmByMlir)
if __name__ == "__main__":
    run_tests()
