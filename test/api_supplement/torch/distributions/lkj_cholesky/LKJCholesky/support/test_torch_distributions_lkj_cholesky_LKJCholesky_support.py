# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributions.lkj_cholesky.LKJCholesky.support 接口功能正确性
API 名称：torch.distributions.lkj_cholesky.LKJCholesky.support
API 签名：support 类级属性 = constraints.corr_cholesky
         （上游源码 torch/distributions/lkj_cholesky.py L64；约束语义见
          torch/distributions/constraints.py L584-L598：下三角、对角为正、
          每行单位范数）

覆盖维度表：
| 覆盖维度     | 说明                                      | 覆盖情况                               |
|--------------|-------------------------------------------|----------------------------------------|
| 空/非空      | 无空值语义（约束对象）                    | 不适用                                 |
| 枚举选项     | support 取值唯一（corr_cholesky），event_ | 已覆盖（test_support_is_corr_cholesky）|
|              | dim 恒为 2                                |                                        |
| 参数类型     | 属性为 constraints.Constraint 实例        | 已覆盖（test_support_is_corr_cholesky）|
| 传参与不传参 | 无参类级属性，仅默认读取                  | 已覆盖（各用例均直接读属性）           |
| 等价类/边界  | 合法（单位行范数）vs 非法（行范数!=1、非  | 已覆盖（test_support_check_accepts、   |
|              | 下三角）两类样本                          | test_support_check_rejects_*）         |
| 正常传参     | 合法相关阵 Cholesky 因子通过 support.check| 已覆盖（test_support_check_accepts）   |
| 异常传参     | 非法因子被 support.check 判 False（而非   | 已覆盖（test_support_check_rejects_    |
|              | 抛错，check 本身无异常路径）              | norm、test_support_check_rejects_tri） |
| 混合设备类型 | 纯 Python 约束契约，check 输入张量统一经  | 不适用（约束语义与设备无关）           |
|              | self.device 构造，不涉及跨设备混用        |                                        |

未覆盖项及原因：
- 无（上述 8 维中不适用项均已给出原因，其余均有具体用例与断言对应）。

注意：断言全部落在「约束对象同一性（is constraints.corr_cholesky）、
     check 对合法/非法样本的布尔判定、event_dim=2」三类可证伪契约上；
     非法样本使用行范数!=1 与含上三角元素两个构造，均能稳定触发 False。
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
class TestLKJCholeskySupport(TestCase):
    """Test cases for torch.distributions.lkj_cholesky.LKJCholesky.support."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_support_is_corr_cholesky(self):
        """support 与 constraints.corr_cholesky 同一，event_dim 恒为 2。"""
        dist = torch.distributions.LKJCholesky(3, 1.0)
        self.assertIs(dist.support, constraints.corr_cholesky)
        self.assertIs(torch.distributions.LKJCholesky.support,
                      constraints.corr_cholesky)
        self.assertEqual(dist.support.event_dim, 2)
        # batched 构造下 support 不变（类级属性）
        batched = torch.distributions.LKJCholesky(
            3, torch.tensor([1.0, 2.0], device=self.device))
        self.assertIs(batched.support, constraints.corr_cholesky)

    def test_support_check_accepts(self):
        """合法样本（单位行范数下三角）必须整体通过 check。"""
        corr = torch.full((3, 3), 0.3, device=self.device)
        idx = torch.arange(3, device=self.device)
        corr[idx, idx] = 1.0
        valid = torch.stack([
            torch.eye(3, device=self.device),
            torch.linalg.cholesky(corr),
        ])
        check = torch.distributions.LKJCholesky.support.check(valid)
        self.assertEqual(check.dtype, torch.bool)
        self.assertTrue(check.all())

    def test_support_check_rejects_norm(self):
        """行范数!=1 的下三角矩阵必须被 check 判为 False（第二行范数=2）。"""
        invalid = torch.zeros(3, 3, device=self.device)
        invalid[0, 0] = 1.0
        invalid[1, 1] = 2.0
        invalid[2, 2] = 1.0
        check = torch.distributions.LKJCholesky.support.check(invalid)
        self.assertFalse(check)

    def test_support_check_rejects_tri(self):
        """含上三角元素的矩阵必须被 check 判为 False（非下三角）。"""
        invalid = torch.eye(3, device=self.device)
        invalid[0, 1] = 0.5
        check = torch.distributions.LKJCholesky.support.check(invalid)
        self.assertFalse(check)

    def test_support_check_batched(self):
        """批输入：check 逐矩阵返回布尔，合法批全 True、混入非法则对应位置 False。"""
        corr = torch.full((3, 3), 0.3, device=self.device)
        idx = torch.arange(3, device=self.device)
        corr[idx, idx] = 1.0
        valid = torch.linalg.cholesky(corr)
        invalid = torch.zeros(3, 3, device=self.device)
        invalid[0, 0] = 1.0
        invalid[1, 1] = 2.0
        invalid[2, 2] = 1.0
        stacked = torch.stack([valid, invalid])
        check = torch.distributions.LKJCholesky.support.check(stacked)
        self.assertEqual(check.shape, torch.Size([2]))
        self.assertTrue(check[0])
        self.assertFalse(check[1])


if __name__ == "__main__":
    run_tests()
