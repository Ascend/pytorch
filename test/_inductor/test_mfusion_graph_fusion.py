# Owner(s): ["module: inductor"]
from unittest import skip, skipIf

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)


try:
    import mfusion  # noqa: F401

    HAS_AKG_MFUSION = True
except ModuleNotFoundError:
    HAS_AKG_MFUSION = False

try:
    from torch_npu._inductor.mfusion.graph_fusion import MFusionPatch
except ImportError:
    MFusionPatch = None  # type: ignore[misc, assignment]

# Gate omits AKG mfusion; keep module importable even if torch_npu mfusion tree is absent.
HAS_MFUSION_STACK = HAS_AKG_MFUSION and MFusionPatch is not None


@skipIf(
    not HAS_MFUSION_STACK,
    "AKG mfusion and torch_npu._inductor.mfusion.graph_fusion not available",
)
class TestMfusionByGraphFusion(TestCase):
    def _compile_and_check(self, model, inputs, *, atol=1e-3, rtol=1e-3):
        expect = model(*inputs)
        with MFusionPatch():
            mfusion_compiled_model = torch.compile(
                model, backend="inductor", fullgraph=True
            )
            result = mfusion_compiled_model(*inputs)
            self.assertEqual(expect, result, atol=atol, rtol=rtol)

    def test_basic_partitioning(self):
        # 不涉及维度broadcast，a, b, c shape完全一致
        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                add = a + b
                mul = add * c
                return mul

        shape = (2, 4)
        a = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        b = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        c = torch.normal(0, 0.1, size=shape, dtype=torch.float16).npu()
        model = Model()
        self._compile_and_check(model, (a, b, c), atol=1e-3, rtol=1e-3)

    def test_broadcast_elementwise(self):
        # 覆盖 broadcast + 标量常量 + 简单激活
        class Model(torch.nn.Module):
            def forward(self, a, b):
                # b: (1, 1, C) broadcast to a
                x = a + b
                x = torch.relu(x)
                return x * 0.5

        dtype = torch.float16
        a = torch.randn(64, 8, 128, device="npu", dtype=dtype)
        b = torch.randn(1, 1, 128, device="npu", dtype=dtype)
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)

    def test_reduction_and_broadcast(self):
        # 覆盖 reduce/keepdim + broadcast + 常量 eps
        class Model(torch.nn.Module):
            def forward(self, x):
                denom = x.sum(dim=-1, keepdim=True) + 1e-3
                y = x / denom
                return y.mean(dim=1)

        dtype = torch.float16
        x = torch.randn(32, 4, 256, device="npu", dtype=dtype)
        # reduce 会引入更大的误差空间，适当放宽
        self._compile_and_check(Model(), (x,), atol=2e-3, rtol=2e-3)

    def test_linear_gelu_with_params(self):
        # 覆盖 get_attr 参数（weight/bias）+ matmul/add + 激活
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(128, 256, bias=True)

            def forward(self, x):
                y = self.linear(x)
                return torch.nn.functional.gelu(y)

        dtype = torch.float16
        torch.manual_seed(0)
        model = Model().eval().to("npu").to(dtype)
        x = torch.randn(64, 128, device="npu", dtype=dtype)
        self._compile_and_check(model, (x,), atol=2e-3, rtol=2e-3)

    @skip("npu inductor bug")
    def test_conv2d_relu_with_params(self):
        # 覆盖 conv2d（参数 get_attr）+ 激活；典型 CV 图结构
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 8, kernel_size=3, stride=1, padding=1, bias=True
                )

            def forward(self, x):
                y = self.conv(x)
                return torch.relu(y + 0.1)

        dtype = torch.float16
        torch.manual_seed(0)
        model = Model().eval().to("npu").to(dtype)
        x = torch.randn(4, 3, 32, 32, device="npu", dtype=dtype)
        self._compile_and_check(model, (x,), atol=3e-3, rtol=3e-3)

    @skip("npu inductor bug")
    def test_chunk_cat_multi_output_and_transpose(self):
        # 覆盖 list/tuple construct/unpack + cat + 多输出 + stride/contiguous 变化
        class Model(torch.nn.Module):
            def forward(self, x):
                # 让输入先变成非 contiguous，触发更多 layout 相关路径
                xt = x.transpose(0, 1)
                a, b = torch.chunk(xt, 2, dim=-1)
                y1 = torch.sin(a + 1.0)
                y2 = torch.cos(b * 2.0)
                y = torch.cat([y1, y2], dim=-1).contiguous()
                # 返回两个输出，覆盖 output tuple
                return y, y.mean(dim=-1)

        dtype = torch.float16
        x = torch.randn(8, 6, 64, device="npu", dtype=dtype)
        self._compile_and_check(Model(), (x,), atol=3e-3, rtol=3e-3)

    def test_mul(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                return a * b

        a = torch.randn(2, 2, dtype=torch.float16).npu()
        b = torch.randn(2, 2, dtype=torch.float16).npu()
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)

    def test_mul_mul_fusion(self):
        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = a * b
                return c * c

        a = torch.randn(2, 2, dtype=torch.float16).npu()
        b = torch.randn(2, 2, dtype=torch.float16).npu()
        self._compile_and_check(Model(), (a, b), atol=1e-3, rtol=1e-3)


instantiate_parametrized_tests(TestMfusionByGraphFusion)

if __name__ == "__main__":
    run_tests()
