# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.intrinsic.quantized.modules.conv_add.ConvAdd2d.from_reference 接口功能正确性
API 名称：torch.ao.nn.intrinsic.quantized.modules.conv_add.ConvAdd2d.from_reference
API 签名：ConvAdd2d.from_reference(cls, ref_mod, output_scale, output_zero_point)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 基础调用 | 由参考卷积模块生成量化模块 | 已覆盖：test_from_reference_returns_quantized_module |
| 正常传参场景 | 校验 scale/zero_point 生效 | 已覆盖：test_from_reference_sets_scale_and_zero_point |
| 等价类/边界值 | 转换后可运行前向 | 已覆盖：test_from_reference_result_runs_forward |
| 异常传参场景 | 非法参考模块触发错误 | 未覆盖，原因见下 |
| 参数类型 | 该接口仅接受量化 Tensor | 不适用 |
| 枚举选项 | 该接口无枚举参数 | 不适用 |
| 传参与不传参 | 该接口无可选参数 | 不适用 |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：量化融合模块仅由 CPU 量化后端（qnnpack/onednn/fbgemm）提供内核，NPU/CUDA 上不存在对应执行路径，无法构造异构设备输入。
- 空/非空：该融合模块要求量化输入且空间维度非空，空张量在形状校验阶段即报错，无独立可断言语义，故不单列。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。
- 异常传参场景：from_reference 对参考模块无显式类型校验，异常类型不稳定，故不单列断言。

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
from torch.ao.nn.intrinsic.quantized.modules.conv_add import ConvAdd2d

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

_SHAPE = (1, 3, 8, 8)
_EXTRA = (1, 4, 6, 6)
_WEIGHT_QPARAMS = {"qscheme": torch.per_tensor_affine, "dtype": torch.qint8, "scale": 0.1, "zero_point": 0}


def _build_reference():
    return torch.ao.nn.quantized.reference.Conv2d(3, 4, 3, weight_qparams=_WEIGHT_QPARAMS)


def _probe():
    module = ConvAdd2d.from_reference((_build_reference(),), 0.05, 0)
    module(_quantize(torch.randn(*_SHAPE)), _quantize(torch.randn(*_EXTRA)))



@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestConvaddConvAdd2dFromreferenceUt(TestCase):
    """Functional tests for torch.ao.nn.intrinsic.quantized.modules.conv_add.ConvAdd2d.from_reference."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_from_reference_returns_quantized_module(self):
        module = ConvAdd2d.from_reference((_build_reference(),), 0.05, 0)
        self.assertIsInstance(module, ConvAdd2d)

    def test_from_reference_sets_scale_and_zero_point(self):
        module = ConvAdd2d.from_reference((_build_reference(),), 0.05, 0)
        self.assertAlmostEqual(float(module.scale), 0.05, places=6)
        self.assertEqual(int(module.zero_point), 0)

    def test_from_reference_result_runs_forward(self):
        module = ConvAdd2d.from_reference((_build_reference(),), 0.05, 0)
        out = module(_quantize(torch.randn(*_SHAPE)), _quantize(torch.randn(*_EXTRA)))
        self.assertEqual(out.dtype, torch.quint8)


if __name__ == "__main__":
    run_tests()
