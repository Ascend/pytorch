# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.qconfig.get_default_qat_qconfig 接口功能正确性
API 名称：torch.ao.quantization.qconfig.get_default_qat_qconfig
API 签名：get_default_qat_qconfig(backend="x86", version=1)

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                                  |
|---------------|--------------------------------------|-----------------------------------------------------------|
| 空/非空       | backend/version 缺省（用 x86/        | 已覆盖：test_default_backend_and_version /                |
|               | 1）与显式指定                        | test_version_zero_matches_documented                      |
| 枚举选项      | backend 取 fbgemm/x86/qnnpack/       | 已覆盖：test_backend_version_one_matches_documented /     |
|               | onednn；version 取 0/1               | test_version_zero_uses_fake_quantize                      |
| 参数类型      | backend 为 str，version 为 int       | 已覆盖：test_returns_qconfig_instance                     |
| 传参与不传参  | 不传参用默认；显式传 backend/version | 已覆盖：test_default_backend_and_version                  |
| 等价类/边界值 | version=0（FakeQuantize）与          | 已覆盖：test_version_zero_uses_fake_quantize /            |
|               | version=1（FusedMovingAvgObs）       | test_version_one_uses_fused_modules                       |
| 正常传参场景  | 四个后端两版均返回可实例化的 QConfig | 已覆盖：test_backend_version_one_matches_documented /     |
|               |                                      | test_observer_instantiation                               |
| 异常传参场景  | 非法 backend / version 触发          | 已覆盖：test_invalid_backend_raises /                     |
|               | AssertionError                       | test_invalid_version_raises                               |
| 混合设备类型  | 纯 Python 配置工厂，                 | 不适用                                                    |
|               | 不涉及张量或多设备输入               |                                                           |

未覆盖项及原因：
- 混合设备类型：get_default_qat_qconfig 只返回 QConfig 配置对象，不接收张量，无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：get_default_qat_qconfig 为纯 Python 配置工厂，无 CPU/NPU 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证返回的 QConfig 结构、版本差异与后端差异。
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

from torch.ao.quantization.fake_quantize import (
    default_fused_per_channel_wt_fake_quant,
    default_fused_wt_fake_quant,
    default_per_channel_weight_fake_quant,
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
from torch.ao.quantization.qconfig import (
    get_default_qat_qconfig,
    QConfig,
    qconfig_equals,
)

_FBGEMM_QAT_V1 = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True
    ),
    weight=default_fused_per_channel_wt_fake_quant,
)
_QNNPACK_QAT_V1 = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False
    ),
    weight=default_fused_wt_fake_quant,
)
_FBGEMM_QAT_V0 = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True
    ),
    weight=default_per_channel_weight_fake_quant,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationQconfigGetDefaultQatQconfigUt(TestCase):
    """Functional tests for torch.ao.quantization.qconfig.get_default_qat_qconfig."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_qconfig_instance(self):
        """The factory returns a QConfig with activation/weight fields."""
        qconfig = get_default_qat_qconfig("x86")
        self.assertIsInstance(qconfig, QConfig)
        self.assertEqual(qconfig._fields, ("activation", "weight"))

    def test_default_backend_and_version(self):
        """Omitting arguments selects the x86 / version=1 default."""
        self.assertTrue(qconfig_equals(get_default_qat_qconfig(), _FBGEMM_QAT_V1))

    def test_backend_version_one_matches_documented(self):
        """fbgemm/x86 share reduce_range=True; qnnpack uses reduce_range=False."""
        self.assertTrue(qconfig_equals(get_default_qat_qconfig("fbgemm"), _FBGEMM_QAT_V1))
        self.assertTrue(qconfig_equals(get_default_qat_qconfig("x86"), _FBGEMM_QAT_V1))
        self.assertTrue(qconfig_equals(get_default_qat_qconfig("qnnpack"), _QNNPACK_QAT_V1))

    def test_version_zero_matches_documented(self):
        """version=0 uses the unfused FakeQuantize modules instead of the fused ones."""
        self.assertTrue(qconfig_equals(get_default_qat_qconfig("fbgemm", version=0), _FBGEMM_QAT_V0))

    def test_version_one_uses_fused_modules(self):
        """version=1 builds FusedMovingAvgObsFakeQuantize for every backend."""
        for backend in ("fbgemm", "x86", "qnnpack", "onednn"):
            qconfig = get_default_qat_qconfig(backend)
            self.assertIsInstance(qconfig.activation(), FusedMovingAvgObsFakeQuantize)
            self.assertIsInstance(qconfig.weight(), FusedMovingAvgObsFakeQuantize)

    def test_version_zero_uses_fake_quantize(self):
        """version=0 builds plain FakeQuantize modules for every backend."""
        for backend in ("fbgemm", "x86", "qnnpack", "onednn"):
            qconfig = get_default_qat_qconfig(backend, version=0)
            self.assertIsInstance(qconfig.activation(), FakeQuantize)
            self.assertIsInstance(qconfig.weight(), FakeQuantize)

    def test_invalid_backend_raises(self):
        """An unsupported backend name is rejected with AssertionError."""
        with self.assertRaisesRegex(AssertionError, "not supported"):
            get_default_qat_qconfig("definitely_not_a_backend")

    def test_invalid_version_raises(self):
        """A version other than 0/1 is rejected with AssertionError."""
        with self.assertRaisesRegex(AssertionError, "not supported"):
            get_default_qat_qconfig("x86", version=5)


if __name__ == "__main__":
    run_tests()
