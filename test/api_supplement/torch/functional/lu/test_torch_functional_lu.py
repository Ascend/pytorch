# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.functional.lu 接口功能正确性
API 名称：torch.functional.lu
API 签名：torch.functional.lu(A, pivot=True, get_infos=False, out=None)

覆盖维度表：
| 覆盖维度         | 说明                                            | 覆盖情况 |
|------------------|-------------------------------------------------|----------|
| shape            | (m, n) 与批量 (*, m, n)                         | 已覆盖 |
| dtype            | float32 / float64                               | 已覆盖 |
| device           | 正常场景 CPU/NPU 双端执行，CPU 为 Golden        | 已覆盖 |
| 参数枚举         | pivot=True、get_infos=True/False                | 已覆盖 |
| 可选参数         | out= 省略与显式传入                             | 已覆盖 |
| 返回值结构       | get_infos 决定 2/3 元组、pivots 为 int32        | 已覆盖 |
| 引用同一性       | out= 时返回调用者传入的同一对象                 | 已覆盖 |
| 异常路径         | out 长度不符、输入维度不足                      | 已覆盖 |
| 混合设备输入     | 单 Tensor 输入，无混合设备路径                  | 未覆盖（不适用） |

未覆盖项及原因：
- pivot=False 在 CPU 上不受支持（RuntimeError）、NPU/CUDA 语义不同，跨设备不稳定，故不作为异常
  路径用例；稳定异常路径改由 out= 长度不符与输入维度不足覆盖。
- 混合设备类型：单 Tensor 输入，无多 Tensor 设备混用场景。
- torch.lu 为废弃接口，调用处统一抑制其 DeprecationWarning，不影响断言。

注意：非图模式浮点 Tensor 以 CPU 结果为 Golden，设备输出移回 CPU 后用
     torch.testing.assert_close 对照；pivots/整型结果用精确相等断言。
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

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == "npu":
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


def _lu(*args, **kwargs):
    """Call torch.functional.lu while muting its deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.functional.lu(*args, **kwargs)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFunctionalLuUt(TestCase):
    """Functional tests for torch.functional.lu (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_is_same_object_as_torch_lu(self):
        """torch.functional.lu is the same object as torch.lu."""
        self.assertIs(torch.functional.lu, torch.lu)

    def test_cpu_golden_factorization(self):
        """Device LU factors/pivots equal the CPU golden result."""
        a = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        cpu_lu, cpu_piv = _lu(a)
        dev_lu, dev_piv = _lu(a.to(self.device))
        torch.testing.assert_close(dev_lu.cpu(), cpu_lu)
        self.assertTrue(torch.equal(dev_piv.cpu(), cpu_piv))

    def test_returns_two_tuple_without_get_infos(self):
        """Without get_infos the result is a 2-tuple (factor, pivots)."""
        result = _lu(torch.eye(3))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_get_infos_returns_three_tuple(self):
        """get_infos=True appends an info IntTensor as the third element."""
        a = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        lu_factor, pivots, info = _lu(a, get_infos=True)
        self.assertEqual(len((lu_factor, pivots, info)), 3)
        self.assertEqual(info.dtype, torch.int32)
        self.assertEqual(int(info.item()), 0)

    def test_pivots_dtype_and_shape(self):
        """Pivots are int32 with one entry per pivot column (min(m, n))."""
        _, pivots = _lu(torch.randn(5, 3))
        self.assertEqual(pivots.dtype, torch.int32)
        self.assertEqual(tuple(pivots.shape), (3,))

    def test_batched_input(self):
        """Batched input yields batched factor and pivots."""
        a = torch.randn(2, 3, 3)
        lu_factor, pivots = _lu(a)
        self.assertEqual(tuple(lu_factor.shape), (2, 3, 3))
        self.assertEqual(tuple(pivots.shape), (2, 3))

    def test_float64_golden(self):
        """float64 input is factorized matching the CPU golden result."""
        a = torch.randn(4, 4, dtype=torch.float64)
        cpu_lu, cpu_piv = _lu(a)
        dev_lu, dev_piv = _lu(a.to(self.device))
        torch.testing.assert_close(dev_lu.cpu(), cpu_lu)
        self.assertTrue(torch.equal(dev_piv.cpu(), cpu_piv))

    def test_out_returns_same_objects(self):
        """out= is filled and its objects are returned by identity."""
        a = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        out_lu = torch.empty_like(a)
        out_piv = torch.empty(2, dtype=torch.int32)
        result = _lu(a, out=(out_lu, out_piv))
        self.assertIs(result[0], out_lu)
        self.assertIs(result[1], out_piv)
        ref_lu, ref_piv = _lu(a)
        torch.testing.assert_close(out_lu, ref_lu)
        self.assertTrue(torch.equal(out_piv, ref_piv))

    def test_out_length_mismatch_raises(self):
        """A wrong-length out tuple is rejected with TypeError."""
        a = torch.tensor([[4.0, 3.0], [6.0, 3.0]])
        with self.assertRaises(TypeError):
            _lu(a, out=(torch.empty_like(a),))

    def test_insufficient_dim_raises(self):
        """A 1D input is rejected with RuntimeError."""
        with self.assertRaises(RuntimeError):
            _lu(torch.tensor([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    run_tests()
