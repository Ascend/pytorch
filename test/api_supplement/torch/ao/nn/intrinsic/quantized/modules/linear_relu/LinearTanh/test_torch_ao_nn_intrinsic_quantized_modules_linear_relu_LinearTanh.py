# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearTanh 接口功能正确性
API 名称：torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearTanh
API 签名：LinearTanh(in_features, out_features)（类构造/属性校验）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 基础调用 | 构造融合线性模块并校验类型 | 已覆盖：test_construct_returns_expected_type |
| 正常传参场景 | 校验融合的浮点模块类型绑定 | 已覆盖：test_float_module_attribute_is_intrinsic |
| 参数类型 | 该接口仅接受量化 Tensor | 不适用 |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 传参与不传参 | 该接口无可选参数 | 不适用 |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：量化融合模块仅由 CPU 量化后端（qnnpack/onednn/fbgemm）提供内核，NPU/CUDA 上不存在对应执行路径，无法构造异构设备输入。
- 空/非空：该融合模块要求量化输入且空间维度非空，空张量在形状校验阶段即报错，无独立可断言语义，故不单列。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：量化融合模块只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
     故本用例不做 CPU/NPU Golden 对照，仅验证量化后端上的功能语义。
"""
import os
import unittest

import torch

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe the device only when the supplement suite actually runs. On the default
# CI collection path the class-level skipUnless fires first, so no torch_npu
# import or device initialization happens here.
DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

import torch.ao.nn.intrinsic
import torch.ao.nn.intrinsic.quantized
import torch.ao.nn.quantized
import torch.ao.nn.quantized.reference
import torch.ao.quantization
import torch.nn
from torch.ao.nn.intrinsic.quantized.modules.linear_relu import LinearTanh

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The quantized intrinsic modules only provide kernels on the CPU quantized
    backend, so there is no NPU/CUDA execution path to compare against.
    """
    supported = torch.backends.quantized.supported_engines
    for name in _ENGINE_CANDIDATES:
        if name not in supported:
            continue
        torch.backends.quantized.engine = name
        if probe is None:
            return name
        try:
            probe()
        except RuntimeError:
            # This engine has no kernel for the target op; try the next one.
            continue
        return name
    raise unittest.SkipTest("No CPU quantized engine can run the target op on this platform")


def _quantize(x, scale=0.05, zero_point=0):
    return torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)

_CTOR_NAME = "LinearTanh"


def _probe():
    return LinearTanh(4, 3)



@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearreluLinearTanhUt(TestCase):
    """Functional tests for torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearTanh."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(None)

    def test_construct_returns_expected_type(self):
        module = LinearTanh(4, 3)
        self.assertIsInstance(module, LinearTanh)
        self.assertIsInstance(module, torch.ao.nn.quantized.Linear)
        self.assertEqual(module._get_name(), "QuantizedLinearTanh")

    def test_float_module_attribute_is_intrinsic(self):
        self.assertIs(LinearTanh._FLOAT_MODULE, torch.ao.nn.intrinsic.LinearTanh)


if __name__ == "__main__":
    run_tests()
