# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Bilinear.extra_repr 接口功能正确性
API 名称：torch.nn.Bilinear.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对 bias=True/False 两种参数结构分别断言 | 已覆盖（test_extra_repr_default_bias/test_extra_repr_bias_false） |
| 枚举选项 | bias 开关 True/False 两分支精确字符串断言 | 已覆盖（test_extra_repr_default_bias/test_extra_repr_bias_false） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_returns_str） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_extra_repr_default_bias） |
| 等价类/边界值 | in1/in2/out_features 属性变化后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_attributes） |
| 正常传参场景 | repr(module) 以 "Bilinear(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 无入参校验路径，无稳定异常契约 | 未覆盖（无参方法不存在非法传参路径） |
| 混合设备类型 | 设备侧构造的 Bilinear 与 CPU 构造的输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- 异常传参场景不适用：extra_repr 无参数、无校验分支。
- 本文件同时看护别名路径 torch.nn.modules.linear.Bilinear.extra_repr：
  torch.nn.Bilinear 与 torch.nn.modules.linear.Bilinear 为同一类对象，见 identity 断言用例。

注意：extra_repr 为纯字符串格式化方法，断言为精确字符串比较，无浮点对照。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

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
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestBilinearExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Bilinear.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Bilinear and torch.nn.modules.linear.Bilinear are the same class."""
        self.assertIs(torch.nn.Bilinear, torch.nn.modules.linear.Bilinear)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for both bias configurations."""
        m = torch.nn.Bilinear(3, 4, 5)
        self.assertIsInstance(m.extra_repr(), str)
        m_no_bias = torch.nn.Bilinear(3, 4, 5, bias=False)
        self.assertIsInstance(m_no_bias.extra_repr(), str)

    def test_extra_repr_default_bias(self):
        """Default bias=True module reports exact in1/in2/out/bias fields."""
        m = torch.nn.Bilinear(3, 4, 5)
        self.assertEqual(
            m.extra_repr(),
            "in1_features=3, in2_features=4, out_features=5, bias=True",
        )

    def test_extra_repr_bias_false(self):
        """bias=False module reports bias=False in the exact representation."""
        m = torch.nn.Bilinear(3, 4, 5, bias=False)
        self.assertEqual(
            m.extra_repr(),
            "in1_features=3, in2_features=4, out_features=5, bias=False",
        )

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.Bilinear(3, 4, 5)
        self.assertTrue(repr(m).startswith("Bilinear("))
        self.assertIn(m.extra_repr(), repr(m))
        self.assertIn("in1_features=3, in2_features=4", repr(m))

    def test_extra_repr_reflects_mutated_attributes(self):
        """extra_repr reads the current attribute values at call time."""
        m = torch.nn.Bilinear(3, 4, 5)
        self.assertIn("in1_features=3", m.extra_repr())
        m.in1_features = 8
        m.in2_features = 9
        m.out_features = 2
        self.assertEqual(
            m.extra_repr(),
            "in1_features=8, in2_features=9, out_features=2, bias=True",
        )

    def test_extra_repr_device_consistency(self):
        """Device-constructed Bilinear has an identical representation string."""
        cpu_m = torch.nn.Bilinear(3, 4, 5)
        device_m = torch.nn.Bilinear(3, 4, 5, device=self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.weight.device.type, self.device_name)


if __name__ == "__main__":
    run_tests()
