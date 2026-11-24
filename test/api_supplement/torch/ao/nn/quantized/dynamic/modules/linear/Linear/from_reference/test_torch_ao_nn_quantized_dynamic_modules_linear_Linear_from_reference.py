# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_reference 接口功能正确性
API 名称：torch.ao.nn.quantized.dynamic.modules.linear.Linear.from_reference
API 签名：Linear.from_reference(cls, ref_qlinear)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | 参考模块权重/偏置非空 | 已覆盖：test_from_reference_forward_shape |
| 枚举选项 | 权重 dtype 由参考模块决定 | 已覆盖：test_from_reference_uses_reference_weight_dtype |
| 参数类型 | 接受 reference 量化 Linear | 已覆盖：test_from_reference_returns_dynamic_linear |
| 传参与不传参 | 该接口仅有一个必选参数 | 不适用 |
| 等价类/边界值 | in_features/out_features 复制一致 | 已覆盖：test_from_reference_copies_features |
| 正常传参场景 | 前向输出与量化 op 结果一致 | 已覆盖：test_from_reference_matches_quantized_op |
| 异常传参场景 | 传入非参考模块触发 AttributeError | 已覆盖：test_from_reference_rejects_float_module |
| 混合设备类型 | 多 Tensor 异构设备输入 | 未覆盖，原因见下 |

未覆盖项及原因：
- 混合设备类型：动态量化模块仅由 CPU 量化后端（onednn/fbgemm/qnnpack）提供内核，
  NPU/CUDA 上无对应执行路径，无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：动态量化 Linear 只在 CPU 量化后端执行，无 NPU/CUDA 数值对照路径，
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

import torch.ao.nn.quantized.dynamic
import torch.ao.nn.quantized.reference
from torch.ao.nn.quantized.dynamic.modules.linear import Linear

_ENGINE_CANDIDATES = ("onednn", "fbgemm", "qnnpack")
_IN_FEATURES = 4
_OUT_FEATURES = 3
_WEIGHT_QPARAMS = {
    "qscheme": torch.per_tensor_affine,
    "dtype": torch.qint8,
    "scale": 0.1,
    "zero_point": 0,
}


def _use_quantized_engine(probe=None):
    """Select a CPU quantized engine able to run the target op.

    The dynamic quantized Linear only provides kernels on the CPU quantized
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


def _build_reference_linear():
    float_mod = torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES)
    ref_mod = torch.ao.nn.quantized.reference.Linear.from_float(float_mod, _WEIGHT_QPARAMS)
    return ref_mod


def _probe():
    module = Linear.from_reference(_build_reference_linear())
    module(torch.randn(2, _IN_FEATURES))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLinearFromreferenceUt(TestCase):
    """Functional tests for the dynamic quantized Linear.from_reference classmethod."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        _use_quantized_engine(_probe)

    def test_from_reference_returns_dynamic_linear(self):
        module = Linear.from_reference(_build_reference_linear())
        self.assertIsInstance(module, Linear)

    def test_from_reference_uses_reference_weight_dtype(self):
        ref_mod = _build_reference_linear()
        module = Linear.from_reference(ref_mod)
        self.assertEqual(module._packed_params.dtype, torch.qint8)
        self.assertEqual(module._packed_params.dtype, ref_mod.weight_dtype)

    def test_from_reference_copies_features(self):
        module = Linear.from_reference(_build_reference_linear())
        self.assertEqual(module.in_features, _IN_FEATURES)
        self.assertEqual(module.out_features, _OUT_FEATURES)

    def test_from_reference_forward_shape(self):
        module = Linear.from_reference(_build_reference_linear())
        out = module(torch.randn(5, _IN_FEATURES))
        self.assertEqual(tuple(out.shape), (5, _OUT_FEATURES))
        self.assertEqual(out.dtype, torch.float32)

    def test_from_reference_matches_quantized_op(self):
        module = Linear.from_reference(_build_reference_linear())
        x = torch.randn(5, _IN_FEATURES)
        expected = torch.ops.quantized.linear_dynamic(
            x, module._packed_params._packed_params, reduce_range=True
        )
        torch.testing.assert_close(module(x), expected)

    def test_from_reference_rejects_float_module(self):
        with self.assertRaises(AttributeError):
            Linear.from_reference(torch.nn.Linear(_IN_FEATURES, _OUT_FEATURES))


if __name__ == "__main__":
    run_tests()
