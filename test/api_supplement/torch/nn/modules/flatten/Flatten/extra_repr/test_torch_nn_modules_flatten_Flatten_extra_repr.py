# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Flatten.extra_repr 接口功能正确性
API 名称：torch.nn.modules.flatten.Flatten.extra_repr
API 签名：extra_repr(self) -> str

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 入参；默认构造与显式参数构造两种状态精确断言 | 已覆盖（default_dims/custom_dims 两用例） |
| 枚举选项 | start_dim/end_dim 取正数、0、负数多个合法值组合 | 已覆盖（default_dims/custom_dims/negative_end_dim） |
| 参数类型 | 返回值必须为 str 类型 | 已覆盖（test_extra_repr_returns_str） |
| 传参与不传参 | extra_repr 无参；字符串读取构造时保存的 start_dim/end_dim | 已覆盖（test_extra_repr_default_dims/test_extra_repr_custom_dims） |
| 等价类/边界值 | 属性被修改后字符串同步反映 | 已覆盖（test_extra_repr_reflects_mutated_dims） |
| 正常传参场景 | repr(module) 以 "Flatten(" 开头且内嵌 extra_repr 结果 | 已覆盖（test_extra_repr_embedded_in_repr） |
| 异常传参场景 | 无入参校验路径，无稳定异常契约 | 未覆盖（无参方法不存在非法传参路径） |
| 混合设备类型 | 设备侧构造的 Flatten 与 CPU 构造的实例输出字符串一致 | 已覆盖（test_extra_repr_device_consistency） |

未覆盖项及原因：
- 异常传参场景不适用：extra_repr 无参数、无校验分支，不存在可稳定断言的异常路径。
- 本文件看护 torch.nn.modules.flatten.Flatten.extra_repr（正名路径）；
  Flatten 模块对象与 forward 的契约分别在同名模块对象/forward 用例文件中覆盖。
- torch.nn.Flatten 与 torch.nn.modules.flatten.Flatten 为同一类对象，模块对象文件断言。
- 设备构造测试在 NPU/GPU 上真实创建模块；无可用设备时按规范抛 RuntimeError。

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
class TestFlattenExtraRepr(TestCase):
    """Cases for torch.nn.modules.flatten.Flatten.extra_repr (NPU/CUDA)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_extra_repr_returns_str(self):
        """extra_repr returns a str for default and custom constructions."""
        self.assertIsInstance(
            torch.nn.modules.flatten.Flatten().extra_repr(), str
        )
        self.assertIsInstance(
            torch.nn.modules.flatten.Flatten(0, 2).extra_repr(), str
        )

    def test_extra_repr_default_dims(self):
        """Default construction reports start_dim=1, end_dim=-1 exactly."""
        m = torch.nn.modules.flatten.Flatten()
        self.assertEqual(m.extra_repr(), "start_dim=1, end_dim=-1")

    def test_extra_repr_custom_dims(self):
        """Custom dims are reported exactly."""
        m = torch.nn.modules.flatten.Flatten(0, 2)
        self.assertEqual(m.extra_repr(), "start_dim=0, end_dim=2")
        m_kw = torch.nn.modules.flatten.Flatten(start_dim=2, end_dim=3)
        self.assertEqual(m_kw.extra_repr(), "start_dim=2, end_dim=3")

    def test_extra_repr_negative_end_dim(self):
        """Negative end_dim is reported as stored."""
        m = torch.nn.modules.flatten.Flatten(1, -2)
        self.assertEqual(m.extra_repr(), "start_dim=1, end_dim=-2")

    def test_extra_repr_embedded_in_repr(self):
        """repr(module) starts with the class name and embeds extra_repr."""
        m = torch.nn.modules.flatten.Flatten(0, 2)
        self.assertEqual(repr(m), "Flatten(start_dim=0, end_dim=2)")

    def test_extra_repr_reflects_mutated_dims(self):
        """extra_repr reads the current attribute values at call time."""
        m = torch.nn.modules.flatten.Flatten()
        m.start_dim = 3
        m.end_dim = 5
        self.assertEqual(m.extra_repr(), "start_dim=3, end_dim=5")

    def test_extra_repr_device_consistency(self):
        """Device-constructed Flatten has an identical representation string."""
        cpu_m = torch.nn.modules.flatten.Flatten(0, 2)
        device_m = torch.nn.modules.flatten.Flatten(0, 2).to(self.device)
        self.assertEqual(cpu_m.extra_repr(), device_m.extra_repr())
        self.assertEqual(device_m.extra_repr(), "start_dim=0, end_dim=2")


if __name__ == "__main__":
    run_tests()
