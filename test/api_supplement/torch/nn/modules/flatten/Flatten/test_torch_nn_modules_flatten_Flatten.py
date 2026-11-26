# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Flatten 模块对象接口功能正确性
API 名称：torch.nn.modules.flatten.Flatten
API 签名：Flatten(start_dim: int = 1, end_dim: int = -1)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无 Tensor 构造入参；默认与显式参数两种构造状态分别断言属性 | 已覆盖（test_constructor_default_attrs/test_constructor_custom_attrs） |
| 枚举选项 | start_dim/end_dim 取正数、0、负数多个合法值 | 已覆盖（default_attrs/custom_attrs/negative_end_dim） |
| 参数类型 | start_dim/end_dim 必须为 int；实例为 Module 子类且无参数无缓冲 | 已覆盖（test_class_identity_and_base/test_stateless_no_parameters_buffers） |
| 传参与不传参 | start_dim 默认 1、end_dim 默认 -1；显式传参覆盖默认 | 已覆盖（test_constructor_default_attrs/test_constructor_custom_attrs） |
| 等价类/边界值 | 多实例状态独立互不影响；同一参数重复构造结果一致 | 已覆盖（test_instances_independent/test_construction_reproducible） |
| 正常传参场景 | 可加入 nn.Sequential；repr 精确断言见 extra_repr 用例文件 | 已覆盖（test_sequential_composition） |
| 异常传参场景 | 构造函数对 start_dim/end_dim 无校验，无稳定异常契约 | 未覆盖（构造层无校验路径；forward 层语义错误在 forward 用例文件覆盖） |
| 混合设备类型 | 模块对象文件不做设备侧 forward 执行 | 未覆盖（设备侧执行与 CPU Golden 见 forward 用例文件） |

未覆盖项及原因：
- 异常传参场景：Flatten.__init__ 不校验参数合法性，构造层不存在稳定异常路径；
  start_dim > end_dim 等前向语义错误归由 Flatten.forward 用例文件覆盖。
- 本文件看护模块对象 API（构造、属性、状态与组合契约）；
  设备侧执行、数值语义与 repr 精确断言分别在 forward / extra_repr 用例文件中覆盖。
- torch.nn.Flatten 与 torch.nn.modules.flatten.Flatten 为同一类对象，
  test_class_identity_and_base 断言其同一性。
- 设备侧调用在 NPU/GPU 上真实执行；无可用设备时按规范抛 RuntimeError。

注意：模块对象只断言构造与状态契约；设备侧 forward 执行与 CPU Golden 对照、
     repr 精确断言分别由 forward / extra_repr 用例文件负责。
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
class TestFlattenModule(TestCase):
    """Module-object contract for torch.nn.modules.flatten.Flatten."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_class_identity_and_base(self):
        """Public alias identity and Module subclassing."""
        self.assertIs(torch.nn.Flatten, torch.nn.modules.flatten.Flatten)
        self.assertTrue(
            issubclass(torch.nn.modules.flatten.Flatten, torch.nn.Module)
        )

    def test_constructor_default_attrs(self):
        """Default construction stores start_dim=1 and end_dim=-1."""
        m = torch.nn.modules.flatten.Flatten()
        self.assertEqual(m.start_dim, 1)
        self.assertEqual(m.end_dim, -1)

    def test_constructor_custom_attrs(self):
        """Explicit construction stores the given dims."""
        m = torch.nn.modules.flatten.Flatten(0, 2)
        self.assertEqual(m.start_dim, 0)
        self.assertEqual(m.end_dim, 2)
        m_kw = torch.nn.modules.flatten.Flatten(start_dim=2, end_dim=3)
        self.assertEqual(m_kw.start_dim, 2)
        self.assertEqual(m_kw.end_dim, 3)

    def test_constructor_negative_end_dim(self):
        """Negative end_dim is accepted and stored as given."""
        m = torch.nn.modules.flatten.Flatten(1, -2)
        self.assertEqual(m.start_dim, 1)
        self.assertEqual(m.end_dim, -2)

    def test_stateless_no_parameters_buffers(self):
        """Flatten owns no parameters and no buffers."""
        m = torch.nn.modules.flatten.Flatten()
        self.assertEqual(len(list(m.parameters())), 0)
        self.assertEqual(len(list(m.buffers())), 0)

    def test_instances_independent(self):
        """Two instances with different dims do not share state."""
        m1 = torch.nn.modules.flatten.Flatten(0, 2)
        m2 = torch.nn.modules.flatten.Flatten(1, -1)
        self.assertEqual((m1.start_dim, m1.end_dim), (0, 2))
        self.assertEqual((m2.start_dim, m2.end_dim), (1, -1))
        m1.start_dim = 3
        self.assertEqual(m2.start_dim, 1)

    def test_construction_reproducible(self):
        """Repeated construction with identical args yields identical state."""
        m1 = torch.nn.modules.flatten.Flatten(2, 4)
        m2 = torch.nn.modules.flatten.Flatten(2, 4)
        self.assertEqual((m1.start_dim, m1.end_dim), (m2.start_dim, m2.end_dim))

    def test_sequential_composition(self):
        """Flatten can be composed inside nn.Sequential."""
        seq = torch.nn.Sequential(
            torch.nn.modules.flatten.Flatten(),
        )
        self.assertIsInstance(seq[0], torch.nn.modules.flatten.Flatten)


if __name__ == "__main__":
    run_tests()
