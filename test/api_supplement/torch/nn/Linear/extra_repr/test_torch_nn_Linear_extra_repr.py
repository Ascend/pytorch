# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.Linear.extra_repr 接口功能正确性
API 名称：torch.nn.Linear.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；对 bias=True/False 两种非空参数结构分别断言 | 已覆盖（test_extra_repr_default_bias/test_extra_repr_bias_false） |
| 枚举选项 | bias 开关 True/False 两个分支均精确断言字符串 | 已覆盖（test_extra_repr_default_bias/test_extra_repr_bias_false） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_returns_str） |
| 传参与不传参 | 无参方法，仅默认调用 | 已覆盖（test_extra_repr_default_bias） |
| 等价类/边界值 | 属性值变化后字符串同步反映（in_features/out_features 变更） | 已覆盖（test_extra_repr_reflects_mutated_attributes） |
| 正常传参场景 | repr(module) 以 "Linear(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 无入参校验路径，无稳定异常契约 | 未覆盖（无参方法不存在非法传参路径） |
| 混合设备类型 | 设备侧构造的 Linear 与 CPU 构造的 Linear 输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- 异常传参场景不适用：extra_repr 无参数、无校验分支，不存在可稳定断言的异常路径。
- 本文件同时看护别名路径 torch.nn.modules.linear.Linear.extra_repr：torch.nn.Linear 与
  torch.nn.modules.linear.Linear 为同一类对象，test_dual_import_paths_same_object 断言其同一性。
- 设备构造测试在 NPU 上真实分配参数张量；无设备（NPU/GPU 均不可用）时按规范抛 RuntimeError。

注意：extra_repr 为纯字符串格式化方法，返回值断言为精确字符串比较；
     设备侧仅验证构造与表示行为，无浮点数值输出，不涉及浮点 Golden 对照。
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
class TestLinearExtraRepr(TestCase):
    """Comprehensive cases for torch.nn.Linear.extra_repr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_dual_import_paths_same_object(self):
        """torch.nn.Linear and torch.nn.modules.linear.Linear are the same class."""
        self.assertIs(torch.nn.Linear, torch.nn.modules.linear.Linear)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for both bias configurations."""
        m = torch.nn.Linear(3, 5)
        self.assertIsInstance(m.extra_repr(), str)
        m_no_bias = torch.nn.Linear(3, 5, bias=False)
        self.assertIsInstance(m_no_bias.extra_repr(), str)

    def test_extra_repr_default_bias(self):
        """Default bias=True module reports exact in/out/bias fields."""
        m = torch.nn.Linear(3, 5)
        self.assertEqual(
            m.extra_repr(), "in_features=3, out_features=5, bias=True"
        )

    def test_extra_repr_bias_false(self):
        """bias=False module reports bias=False in the exact representation."""
        m = torch.nn.Linear(3, 5, bias=False)
        self.assertEqual(
            m.extra_repr(), "in_features=3, out_features=5, bias=False"
        )

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.Linear(3, 5)
        self.assertEqual(
            repr(m), "Linear(in_features=3, out_features=5, bias=True)"
        )
        m_no_bias = torch.nn.Linear(3, 5, bias=False)
        self.assertEqual(
            repr(m_no_bias), "Linear(in_features=3, out_features=5, bias=False)"
        )

    def test_extra_repr_reflects_mutated_attributes(self):
        """extra_repr reads the current attribute values at call time."""
        m = torch.nn.Linear(3, 5)
        self.assertIn("in_features=3", m.extra_repr())
        m.in_features = 8
        m.out_features = 2
        self.assertEqual(
            m.extra_repr(), "in_features=8, out_features=2, bias=True"
        )

    def test_extra_repr_device_consistency(self):
        """Device-constructed Linear has an identical representation string."""
        cpu_m = torch.nn.Linear(4, 6)
        device_m = torch.nn.Linear(4, 6).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.weight.device.type, self.device_name)
        device_m_direct = torch.nn.Linear(4, 6, device=self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m_direct.extra_repr())


if __name__ == "__main__":
    run_tests()
