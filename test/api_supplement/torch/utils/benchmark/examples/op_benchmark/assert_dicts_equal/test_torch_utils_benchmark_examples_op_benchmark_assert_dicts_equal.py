# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.benchmark.examples.op_benchmark.assert_dicts_equal 接口功能正确性
API 名称：torch.utils.benchmark.examples.op_benchmark.assert_dicts_equal
API 签名：torch.utils.benchmark.examples.op_benchmark.assert_dicts_equal(dict_0, dict_1)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空字典与非空字典输入                 | 已覆盖（empty_dicts_raises、equal_dicts）          |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 标量、字符串、numpy 数组与 dtype 值  | 已覆盖（equal_dicts、equal_numpy_arrays、ignore_dtype_key） |
| 传参与不传参     | 两个必选字典传参与缺参               | 已覆盖（equal_dicts、missing_arg_raises）          |
| 等价类/边界值    | 忽略 dtype 键、空字典、缺键边界      | 已覆盖（ignore_dtype_key、empty_dicts_raises、missing_key_raises） |
| 正常传参场景     | 等价普通字典与 numpy 数组字典        | 已覆盖（equal_dicts、equal_numpy_arrays、ignore_dtype_key） |
| 异常传参场景     | 值不等、缺键、空字典及缺参报错       | 已覆盖（different_values_raises、different_numpy_arrays_raises 等） |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 枚举选项：接口仅比较两个 dict 参数，无枚举选项。
- 混合设备类型：接口用于比较基准配置字典，不涉及多设备 Tensor 计算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import numpy as np
import torch
from torch.utils.benchmark.examples.op_benchmark import assert_dicts_equal

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
class TestAssertDictsEqual(TestCase):
    """Test cases for torch.utils.benchmark.examples.op_benchmark.assert_dicts_equal."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_equal_dicts(self):
        """Identical scalar and string dicts pass and return None."""
        d0 = {"k0": 16, "name": "unary", "contiguous": True}
        d1 = {"k0": 16, "name": "unary", "contiguous": True}
        ret = assert_dicts_equal(d0, d1)
        self.assertIsNone(ret)

    def test_equal_numpy_arrays(self):
        """Dicts containing equal numpy arrays pass without ValueError."""
        d0 = {"order": np.array([0, 1, 2]), "steps": np.ones((2, 1))}
        d1 = {"order": np.array([0, 1, 2]), "steps": np.ones((2, 1))}
        ret = assert_dicts_equal(d0, d1)
        self.assertIsNone(ret)

    def test_ignore_dtype_key(self):
        """Different values under the 'dtype' key are ignored when other keys match."""
        d0 = {"dtype": torch.float32, "size": 64}
        d1 = {"dtype": torch.int32, "size": 64}
        ret = assert_dicts_equal(d0, d1)
        self.assertIsNone(ret)

    def test_different_values_raises(self):
        """Differing values for non-dtype keys raise AssertionError."""
        d0 = {"a": 1}
        d1 = {"a": 2}
        with self.assertRaises(AssertionError):
            assert_dicts_equal(d0, d1)

    def test_different_numpy_arrays_raises(self):
        """Differing numpy array values raise AssertionError."""
        d0 = {"arr": np.ones((2, 2))}
        d1 = {"arr": np.zeros((2, 2))}
        with self.assertRaises(AssertionError):
            assert_dicts_equal(d0, d1)

    def test_missing_key_raises(self):
        """Key present in dict_0 but missing in dict_1 raises KeyError."""
        with self.assertRaises(KeyError):
            assert_dicts_equal({"a": 1}, {"b": 1})

    def test_empty_dicts_raises(self):
        """Empty dicts vacuously satisfy all(...) and raise AssertionError."""
        with self.assertRaises(AssertionError):
            assert_dicts_equal({}, {})

    def test_missing_arg_raises(self):
        """Calling assert_dicts_equal with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            assert_dicts_equal({"a": 1})


if __name__ == "__main__":
    run_tests()
