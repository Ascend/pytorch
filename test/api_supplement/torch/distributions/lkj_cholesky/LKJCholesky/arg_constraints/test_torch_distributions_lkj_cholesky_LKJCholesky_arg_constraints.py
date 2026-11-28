# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints 接口
         功能正确性
API 名称：torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints
API 签名：arg_constraints 类级字典属性
         {"concentration": constraints.positive}
         （上游源码 torch/distributions/lkj_cholesky.py L62-L63）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 约束字典非空，键集恰为单键                | 已覆盖（test_arg_constraints_keys）    |
| 枚举选项     | 约束键集的枚举（仅 concentration 一键）   | 已覆盖（test_arg_constraints_keys）    |
| 参数类型     | 属性为 dict，值为 constraints.Constraint  | 已覆盖（test_arg_constraints_keys）    |
| 传参与不传参 | 无参类级属性，仅默认读取                  | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | concentration=0 边界（positive 不含 0）与 | 已覆盖（test_concentration_constraint_ |
|              | 负值等价类                                | semantics）                            |
| 正常传参     | 正值满足约束；validate_args=True 时合法   | 已覆盖（test_concentration_constraint_ |
|              | concentration 构造成功                   | semantics、test_valid_construction）   |
| 异常传参     | validate_args=True 时负 concentration 与  | 已覆盖（test_invalid_concentration_    |
|              | dim<2 的构造均抛 ValueError               | raises、test_dim_below_two_raises）    |
| 混合设备类型 | 纯 Python 约束契约，check 输入张量统一经  | 不适用（约束语义与设备无关）           |
|              | self.device 构造，不涉及跨设备混用        |                                        |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：断言全部落在「键集与约束对象同一性（is constraints.positive）、
     positive.check 的成员/边界语义、validate_args=True 构造期校验」三类
     可证伪契约上。
"""
import os
import unittest

import torch
import torch.distributions  # noqa: F401  (被测 API 所在包)
from torch.distributions import constraints

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# 仅在补充测试真正运行（RUN_FLAG）时才探测设备：默认 CI 收集会被类级
# skipUnless 跳过，此时不做 torch_npu 导入与 NPU/CUDA 设备初始化，避免
# 不必要的探测开销；被跳过时 DEVICE_NAME 仅为占位值 'cpu'。
DEVICE_NAME = 'cpu'
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = 'npu'
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

# ORDER CONSTRAINT: import torch_npu.testing.testcase BEFORE
# torch_npu.contrib.transfer_to_npu (patching first breaks the import).
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    # No torch_npu (e.g. a plain CUDA/CPU box). Fall back to the stdlib
    # TestCase. This file does not compare floating point tensors, so the
    # fallback is safe here.
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

# transfer_to_npu rebinds torch.cuda.* and torch.Event to the NPU
# implementations, so the test body uses generic entry points unmodified
# on NPU and CUDA. Keep AFTER the testcase import above.
if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestLKJCholeskyArgConstraints(TestCase):
    """Test cases for torch.distributions.lkj_cholesky.LKJCholesky.arg_constraints."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_arg_constraints_keys(self):
        """键集恰为 {concentration}，且约束对象与 constraints.positive 同一。"""
        arg_constraints = torch.distributions.LKJCholesky.arg_constraints
        self.assertEqual(set(arg_constraints.keys()), {"concentration"})
        self.assertIs(arg_constraints["concentration"], constraints.positive)

    def test_concentration_constraint_semantics(self):
        """positive.check 的成员语义：正值通过，负值与 0 边界不通过。"""
        check = constraints.positive.check
        positive_vals = torch.tensor([0.5, 2.0, 1e-6], device=self.device)
        self.assertEqual(check(positive_vals).dtype, torch.bool)
        self.assertTrue(check(positive_vals).all())
        non_positive_vals = torch.tensor([-1.0, 0.0, -1e-6], device=self.device)
        self.assertFalse(check(non_positive_vals).any())

    def test_valid_construction(self):
        """validate_args=True 时合法 concentration 构造成功且维度正确。"""
        dist = torch.distributions.LKJCholesky(
            3, torch.tensor(1.5, device=self.device), validate_args=True)
        self.assertEqual(dist.batch_shape, torch.Size([]))
        self.assertEqual(dist.event_shape, torch.Size([3, 3]))
        torch.testing.assert_close(
            dist.concentration, torch.tensor(1.5, device=self.device))

    def test_invalid_concentration_raises(self):
        """validate_args=True 时负 concentration 违反 arg_constraints，抛 ValueError。"""
        with self.assertRaises(ValueError):
            torch.distributions.LKJCholesky(
                3, torch.tensor(-1.0, device=self.device), validate_args=True)

    def test_dim_below_two_raises(self):
        """dim<2 的构造抛 ValueError（源码 L72-L75 的前置校验）。"""
        with self.assertRaises(ValueError):
            torch.distributions.LKJCholesky(1, 1.0)


if __name__ == "__main__":
    run_tests()
