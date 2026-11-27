# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.ao.quantization.utils.check_min_max_valid 接口功能正确性
API 名称：torch.ao.quantization.utils.check_min_max_valid
API 签名：check_min_max_valid(min_val, max_val) -> bool

覆盖维度表：
| 覆盖维度      | 说明                                         | 覆盖情况                                          |
|---------------|----------------------------------------------|---------------------------------------------------|
| 空/非空       | 空张量（numel==0）/ 非空张量                 | 已覆盖：test_empty_tensors_false_with_warning、   |
|               |                                              | test_valid_scalar_true                            |
| 枚举选项      | 标量张量 / 向量张量两种形态                  | 已覆盖：test_valid_scalar_true、                  |
|               |                                              | test_valid_vector_true                            |
| 参数类型      | min_val/max_val 为 torch.Tensor              | 已覆盖：test_valid_scalar_true                    |
| 传参与不传参  | 该接口无可选参数                             | 不适用                                            |
| 等价类/边界值 | min==max、min>max（标量与向量）、inf/-inf    | 已覆盖：test_equal_min_max_true、                 |
|               |                                              | test_min_greater_than_max_scalar_raises、         |
|               |                                              | test_min_greater_than_max_vector_raises、         |
|               |                                              | test_inf_scalar_false_with_warning                |
| 正常传参场景  | 合法 min<max 返回 True                       | 已覆盖：test_valid_scalar_true、                  |
|               |                                              | test_valid_vector_true                            |
| 异常传参场景  | min>max 触发 AssertionError；mw 空/inf 返回  | 已覆盖：test_min_greater_than_max_scalar_raises、 |
|               | False                                        |                                                   |
|               | 且告警                                       | test_empty_tensors_false_with_warning             |
| 混合设备类型  | 多 Tensor 异构设备输入                       | 未覆盖，见下                                      |

未覆盖项及原因：
- 传参与不传参：接口仅两个必填参数，无可选参数可覆盖。
- 混合设备类型：校验只在 CPU 上进行数值比较，无 NPU/CUDA 参与路径，构造异构设备输入无意义，
  故不覆盖。

注：该 API 为纯 Python 校验工具（框架类 A），不产生 Tensor 数值输出，按布尔/异常/告警断言，
     不做 CPU Golden 对照。
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

from torch.ao.quantization.utils import check_min_max_valid


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCheckMinMaxValidUt(TestCase):
    """Functional tests for torch.ao.quantization.utils.check_min_max_valid."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_valid_scalar_true(self):
        self.assertTrue(check_min_max_valid(torch.tensor(0.0), torch.tensor(1.0)))

    def test_valid_vector_true(self):
        min_val = torch.tensor([-1.0, 0.0, 2.0])
        max_val = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(check_min_max_valid(min_val, max_val))

    def test_equal_min_max_true(self):
        # min == max is valid (only strict min > max is rejected).
        self.assertTrue(check_min_max_valid(torch.tensor(0.0), torch.tensor(0.0)))

    def test_empty_tensors_false_with_warning(self):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = check_min_max_valid(torch.tensor([]), torch.tensor([]))
        self.assertFalse(result)
        self.assertEqual(len(records), 1)
        self.assertTrue(issubclass(records[0].category, UserWarning))

    def test_inf_scalar_false_with_warning(self):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            result = check_min_max_valid(torch.tensor(float("inf")), torch.tensor(float("-inf")))
        self.assertFalse(result)
        self.assertEqual(len(records), 1)

    def test_min_greater_than_max_scalar_raises(self):
        with self.assertRaises(AssertionError):
            check_min_max_valid(torch.tensor(2.0), torch.tensor(1.0))

    def test_min_greater_than_max_vector_raises(self):
        min_val = torch.tensor([0.0, 5.0])
        max_val = torch.tensor([1.0, 4.0])
        with self.assertRaises(AssertionError):
            check_min_max_valid(min_val, max_val)


if __name__ == "__main__":
    run_tests()
