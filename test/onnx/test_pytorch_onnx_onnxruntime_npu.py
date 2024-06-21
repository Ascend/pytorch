# Owner(s): ["module: onnx"]

import unittest

import onnx_test_common

import onnxruntime  # noqa: F401
import parameterized

import torch
from onnx_test_common import MAX_ONNX_OPSET_VERSION, MIN_ONNX_OPSET_VERSION
from pytorch_test_common import (
    skipIfNoBFloat16NPU,
    skipIfNoNPU,
    skipIfUnsupportedMinOpsetVersion,
    skipScriptTest,
)
from test_pytorch_onnx_onnxruntime import _parameterized_class_attrs_and_values
from torch.npu.amp import autocast
from torch.testing._internal import common_utils
import torch_npu
import torch_npu.testing


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(
        MIN_ONNX_OPSET_VERSION, MAX_ONNX_OPSET_VERSION
    ),
    class_name_func=onnx_test_common.parameterize_class_name,
)
class TestONNXRuntime_npu(onnx_test_common._TestONNXRuntime):
    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoNPU
    def test_gelu_fp16(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        x = torch.randn(
            2,
            4,
            5,
            6,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("npu"),
        )
        self.run_test(GeluModel(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipIfNoNPU
    @skipScriptTest()
    def test_layer_norm_fp16(self):
        class LayerNormModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = torch.nn.LayerNorm([10, 10])

            @autocast()
            def forward(self, x):
                return self.layer_norm(x)

        x = torch.randn(
            20,
            5,
            10,
            10,
            requires_grad=True,
            dtype=torch.float16,
            device=torch.device("npu"),
        )
        self.run_test(LayerNormModel().npu(), x, rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipIfNoNPU
    @skipScriptTest()
    def test_softmaxCrossEntropy_fusion_fp16(self):
        class FusionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction="none")
                self.m = torch.nn.LogSoftmax(dim=1)

            @autocast()
            def forward(self, input_, target):
                output = self.loss(self.m(2 * input_), target)
                return output

        N, C = 5, 4
        input_ = torch.randn(N, 16, dtype=torch.float16, device=torch.device("npu"))
        target = torch.empty(N, dtype=torch.long, device=torch.device("npu")).random_(
            0, C
        )

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(FusionModel(), (input_, target))

    @skipIfNoNPU
    @skipScriptTest()
    def test_apex_o2(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)

        try:
            from apex import amp
        except Exception as e:
            raise unittest.SkipTest("Apex is not available") from e
        input_ = torch.randn(3, 3, device=torch.device("npu"))
        model = LinearModel().npu()
        model = amp.initialize(model, opt_level="O2")
        self.run_test(model, input_)

    # ONNX supports bfloat16 for opsets >= 13
    # Add, Sub and Mul ops don't support bfloat16 cpu in onnxruntime.
    @skipIfUnsupportedMinOpsetVersion(13)
    @skipIfNoBFloat16NPU
    def test_arithmetic_bfp16(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4, dtype=torch.bfloat16, device=torch.device("npu"))
                x = x.type_as(y)
                return torch.mul(torch.add(x, y), torch.sub(x, y)).to(
                    dtype=torch.float16
                )

        x = torch.ones(
            3, 4, requires_grad=True, dtype=torch.float16, device=torch.device("npu")
        )
        self.run_test(MyModule(), x, rtol=1e-3, atol=1e-5)

    @skipIfNoNPU
    def test_deduplicate_initializers_diff_devices(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(
                    torch.ones(2, 3, device=torch.device("cpu"))
                )
                self.b = torch.nn.Parameter(torch.ones(3, device=torch.device("npu")))

            def forward(self, x, y):
                return torch.matmul(self.w, x), y + self.b

        x = torch.randn(3, 3, device=torch.device("cpu"))
        y = torch.randn(3, 3, device=torch.device("npu"))
        self.run_test(Model(), (x, y))


if __name__ == "__main__":
    common_utils.run_tests()
