# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Unflatten 模块对象接口功能正确性
API 名称：torch.nn.modules.flatten.Unflatten
API 签名：Unflatten(dim: int, unflattened_size: Union[torch.Size, Tuple, List])

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | unflattened_size 支持 tuple/list/torch.Size 三种形态与 dim 正负值 | 已覆盖（tuple_size/list_size/size_obj/negative_dim） |
| 枚举选项 | dim 取正数、0、负数；unflattened_size 取三种容器类型 | 已覆盖（tuple_size/list_size/size_obj/negative_dim/dim_zero） |
| 参数类型 | 非法元素（float/str）与非容器（int）触发 TypeError | 已覆盖（invalid_element_type/non_tuple_size/invalid_str_element） |
| 传参与不传参 | dim/unflattened_size 均必填，缺省构造断言 TypeError | 已覆盖（test_constructor_requires_args） |
| 等价类/边界值 | 多实例状态独立互不影响；无参数无缓冲；构造保存原容器对象 | 已覆盖（stateless/independent/stored_as_passed 三用例） |
| 正常传参场景 | 可加入 nn.Sequential；repr 精确断言见 extra_repr 用例文件 | 已覆盖（test_sequential_composition） |
| 异常传参场景 | 非法元素类型/非容器类型触发 TypeError | 已覆盖（invalid_element_type/non_tuple_size/invalid_str_element） |
| 混合设备类型 | 模块对象文件不做设备侧 forward 执行 | 未覆盖（设备侧执行与 CPU Golden 见 forward 用例文件） |

未覆盖项及原因：
- torch.Size 在 2.15 构造时按原样存储（不转换为 tuple），由 test_size_stored_as_passed 精确断言。
- 本文件看护模块对象 API（构造、属性、状态、校验与组合契约）；
  设备侧执行、数值语义与 repr 精确断言分别在 forward / extra_repr 用例文件中覆盖。
- torch.nn.Unflatten 与 torch.nn.modules.flatten.Unflatten 为同一类对象，
  test_class_identity_and_base 断言其同一性。
- 设备侧调用在 NPU/GPU 上真实执行；无可用设备时按规范抛 RuntimeError。

注意：模块对象只断言构造、状态与校验契约；设备侧 forward 执行与 CPU Golden 对照、
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
class TestUnflattenModule(TestCase):
    """Module-object contract for torch.nn.modules.flatten.Unflatten."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_class_identity_and_base(self):
        """Public alias identity and Module subclassing."""
        self.assertIs(torch.nn.Unflatten, torch.nn.modules.flatten.Unflatten)
        self.assertTrue(
            issubclass(torch.nn.modules.flatten.Unflatten, torch.nn.Module)
        )

    def test_constructor_tuple_size(self):
        """Tuple unflattened_size and int dim are stored as given."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        self.assertEqual(m.dim, 1)
        self.assertEqual(m.unflattened_size, (2, 5, 5))

    def test_constructor_list_size(self):
        """List unflattened_size is accepted and stored as a list."""
        m = torch.nn.modules.flatten.Unflatten(0, [4, 3])
        self.assertEqual(m.dim, 0)
        self.assertEqual(m.unflattened_size, [4, 3])

    def test_constructor_size_obj(self):
        """torch.Size unflattened_size is accepted."""
        m = torch.nn.modules.flatten.Unflatten(1, torch.Size([2, 5, 5]))
        self.assertEqual(m.dim, 1)
        self.assertEqual(tuple(m.unflattened_size), (2, 5, 5))

    def test_size_stored_as_passed(self):
        """The unflattened_size attribute preserves the passed container type."""
        m_tuple = torch.nn.modules.flatten.Unflatten(1, (2, 5))
        m_size = torch.nn.modules.flatten.Unflatten(1, torch.Size([2, 5]))
        self.assertIsInstance(m_tuple.unflattened_size, tuple)
        self.assertIsInstance(m_size.unflattened_size, torch.Size)

    def test_constructor_negative_dim(self):
        """Negative dim is accepted and stored as given."""
        m = torch.nn.modules.flatten.Unflatten(-1, (2, 25))
        self.assertEqual(m.dim, -1)
        self.assertEqual(m.unflattened_size, (2, 25))

    def test_constructor_dim_zero(self):
        """dim=0 is accepted and stored as given."""
        m = torch.nn.modules.flatten.Unflatten(0, (2, 25))
        self.assertEqual(m.dim, 0)

    def test_stateless_no_parameters_buffers(self):
        """Unflatten owns no parameters and no buffers."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5))
        self.assertEqual(len(list(m.parameters())), 0)
        self.assertEqual(len(list(m.buffers())), 0)

    def test_instances_independent(self):
        """Two instances with different args do not share state."""
        m1 = torch.nn.modules.flatten.Unflatten(1, (2, 5))
        m2 = torch.nn.modules.flatten.Unflatten(0, (3, 4))
        self.assertEqual((m1.dim, m1.unflattened_size), (1, (2, 5)))
        self.assertEqual((m2.dim, m2.unflattened_size), (0, (3, 4)))
        m1.dim = 2
        self.assertEqual(m2.dim, 0)

    def test_invalid_element_type_raises(self):
        """A float element in unflattened_size triggers TypeError."""
        with self.assertRaises(TypeError):
            torch.nn.modules.flatten.Unflatten(1, (2.0, 5))

    def test_invalid_str_element_raises(self):
        """A str element in unflattened_size triggers TypeError."""
        with self.assertRaises(TypeError):
            torch.nn.modules.flatten.Unflatten(1, (2, "a"))

    def test_non_tuple_size_raises(self):
        """A non-container unflattened_size (int) triggers TypeError."""
        with self.assertRaises(TypeError):
            torch.nn.modules.flatten.Unflatten(1, 5)

    def test_constructor_requires_args(self):
        """Both dim and unflattened_size are required (TypeError if missing)."""
        with self.assertRaises(TypeError):
            torch.nn.modules.flatten.Unflatten()
        with self.assertRaises(TypeError):
            torch.nn.modules.flatten.Unflatten(1)

    def test_sequential_composition(self):
        """Unflatten can be composed inside nn.Sequential."""
        seq = torch.nn.Sequential(
            torch.nn.modules.flatten.Unflatten(1, (2, 5, 5)),
        )
        self.assertIsInstance(seq[0], torch.nn.modules.flatten.Unflatten)


if __name__ == "__main__":
    run_tests()
