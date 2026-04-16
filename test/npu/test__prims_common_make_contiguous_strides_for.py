# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._prims_common.make_contiguous_strides_for 接口功能正确性
API 名称：torch._prims_common.make_contiguous_strides_for
API 签名：make_contiguous_strides_for(shape: ShapeType, row_major: bool = True) -> tuple[Union[_IntLikeT, int], ...]

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况    |
|------------------|--------------------------------------------------------------|-------------|
| 空/非空          | shape 为空与非空                                             | 已覆盖      |
| 枚举选项         | row_major=True / False                                       | 已覆盖      |
| 参数类型         | tuple[int, ...] / list[int] / bool                           | 已覆盖      |
| 传参与不传参     | row_major 默认 vs 显式传入                                   | 已覆盖      |
| 等价类/边界值    | 0-dim / 1D / 2D / ND / 空 shape                              | 已覆盖      |
| 正常传参场景     | 返回 tuple 类型且长度与 shape 一致                           | 已覆盖      |
| 异常传参场景     | 传入非序列类型触发 TypeError                                 | 已覆盖      |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、返回类型/长度符合预期），
     不做精度和数值正确性校验。
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class TestPrimsCommonMakeContiguousStridesFor(TestCase):
    """Test cases for torch._prims_common.make_contiguous_strides_for."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_make_contiguous_strides_for_basic(self):
        """Basic shape returns correct row-major contiguous strides."""
        result = torch._prims_common.make_contiguous_strides_for((2, 3, 4))
        self.assertTrue(result == (12, 4, 1), f"Expected (12, 4, 1), got {result}")

    def test_make_contiguous_strides_for_row_major_false(self):
        """row_major=False returns correct non-row-major contiguous strides."""
        result = torch._prims_common.make_contiguous_strides_for((2, 3, 4), row_major=False)
        self.assertTrue(result == (12, 1, 3), f"Expected (12, 1, 3), got {result}")

    def test_make_contiguous_strides_for_empty_shape(self):
        """Empty shape returns empty tuple."""
        result = torch._prims_common.make_contiguous_strides_for(())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 0)

    def test_make_contiguous_strides_for_1d(self):
        """1D shape returns tuple of length 1."""
        result = torch._prims_common.make_contiguous_strides_for((5,))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_make_contiguous_strides_for_list_input(self):
        """List input for shape returns tuple."""
        result = torch._prims_common.make_contiguous_strides_for([2, 3])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_make_contiguous_strides_for_invalid_type(self):
        """Invalid shape type raises TypeError."""
        with self.assertRaises(TypeError):
            torch._prims_common.make_contiguous_strides_for("invalid")


if __name__ == "__main__":
    run_tests()
