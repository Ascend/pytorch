# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.qconfig.get_default_qconfig 接口功能正确性
API 名称：torch.ao.quantization.qconfig.get_default_qconfig
API 签名：get_default_qconfig(backend="x86", version=0)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
| --- | --- | --- |
| 空/非空 | backend 缺省（空，用默认 x86）与显式指定四种后端 | 已覆盖：test_default_backend_is_x86 / test_backend_matches_documented_qconfig |
| 枚举选项 | backend 取 fbgemm/x86/qnnpack/onednn | 已覆盖：test_backend_matches_documented_qconfig |
| 参数类型 | backend 为 str，version 为 int | 已覆盖：test_default_backend_is_x86 / test_returns_qconfig_instance |
| 传参与不传参 | 不传参用默认 x86/version=0；显式传 backend/version | 已覆盖：test_default_backend_is_x86 / test_explicit_version_zero |
| 等价类/边界值 | fbgemm 与 x86 等价、qnnpack 与 onednn 使用非 reduce_range | 已覆盖：test_fbgemm_equals_x86 |
| 正常传参场景 | 四个后端均返回可实例化的 QConfig | 已覆盖：test_backend_matches_documented_qconfig / test_observer_instantiation |
| 异常传参场景 | 非法 backend / version 触发 AssertionError | 已覆盖：test_invalid_backend_raises / test_invalid_version_raises |
| 混合设备类型 | 纯 Python 配置工厂，不涉及张量或多设备输入 | 不适用 |

未覆盖项及原因：
- 混合设备类型：get_default_qconfig 只返回 QConfig 配置对象，不接收张量，无法构造异构设备输入。
- 图模式：本用例针对 eager 功能，未覆盖 torch.compile / NPUGraph 图模式路径。

注：get_default_qconfig 为纯 Python 配置工厂，无 CPU/NPU 数值对照路径，
     故不做 CPU/NPU Golden 对照，仅验证返回的 QConfig 结构与后端差异。
"""
import os
import unittest
import warnings

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

from torch.ao.quantization.observer import (
    default_per_channel_weight_observer,
    default_weight_observer,
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.ao.quantization.qconfig import (
    get_default_qconfig,
    QConfig,
    qconfig_equals,
)

_FBGEMM_QCONFIG = QConfig(
    activation=HistogramObserver.with_args(reduce_range=True),
    weight=default_per_channel_weight_observer,
)
_QNNPACK_QCONFIG = QConfig(
    activation=HistogramObserver.with_args(reduce_range=False),
    weight=default_weight_observer,
)
_ONEDNN_QCONFIG = QConfig(
    activation=HistogramObserver.with_args(reduce_range=False),
    weight=default_per_channel_weight_observer,
)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTorchAoQuantizationQconfigGetDefaultQconfigUt(TestCase):
    """Functional tests for torch.ao.quantization.qconfig.get_default_qconfig."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_qconfig_instance(self):
        """The factory returns a QConfig with activation/weight fields."""
        qconfig = get_default_qconfig("x86")
        self.assertIsInstance(qconfig, QConfig)
        self.assertEqual(qconfig._fields, ("activation", "weight"))

    def test_default_backend_is_x86(self):
        """Omitting backend selects the x86 default documented in the signature."""
        self.assertTrue(qconfig_equals(get_default_qconfig(), _FBGEMM_QCONFIG))

    def test_explicit_version_zero(self):
        """Explicit version=0 is the only supported version and matches the default."""
        self.assertTrue(qconfig_equals(get_default_qconfig("x86", version=0), _FBGEMM_QCONFIG))

    def test_backend_matches_documented_qconfig(self):
        """Each supported backend returns the documented QConfig structure."""
        self.assertTrue(qconfig_equals(get_default_qconfig("fbgemm"), _FBGEMM_QCONFIG))
        self.assertTrue(qconfig_equals(get_default_qconfig("x86"), _FBGEMM_QCONFIG))
        self.assertTrue(qconfig_equals(get_default_qconfig("qnnpack"), _QNNPACK_QCONFIG))
        self.assertTrue(qconfig_equals(get_default_qconfig("onednn"), _ONEDNN_QCONFIG))

    def test_fbgemm_equals_x86(self):
        """fbgemm and x86 share the same observer configuration."""
        self.assertTrue(qconfig_equals(get_default_qconfig("fbgemm"), get_default_qconfig("x86")))

    def test_observer_instantiation(self):
        """The activation/weight entries build the expected observer classes."""
        x86 = get_default_qconfig("x86")
        self.assertIsInstance(x86.activation(), HistogramObserver)
        self.assertIsInstance(x86.weight(), PerChannelMinMaxObserver)
        qnnpack = get_default_qconfig("qnnpack")
        self.assertIsInstance(qnnpack.weight(), MinMaxObserver)

    def test_reduce_range_warning_emitted(self):
        """The x86 activation observer warns that reduce_range is deprecated."""
        qconfig = get_default_qconfig("x86")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            qconfig.activation()
        self.assertTrue(any("reduce_range" in str(item.message) for item in caught))

    def test_invalid_backend_raises(self):
        """An unsupported backend name is rejected with AssertionError."""
        with self.assertRaisesRegex(AssertionError, "not supported"):
            get_default_qconfig("definitely_not_a_backend")

    def test_invalid_version_raises(self):
        """A version other than 0 is rejected with AssertionError."""
        with self.assertRaisesRegex(AssertionError, "not supported"):
            get_default_qconfig("x86", version=2)


if __name__ == "__main__":
    run_tests()
