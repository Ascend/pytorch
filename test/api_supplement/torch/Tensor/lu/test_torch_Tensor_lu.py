# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.lu 接口功能正确性
API 名称：torch.Tensor.lu
API 签名：Tensor.lu(pivot=True, get_infos=False) -> (Tensor, Tensor[, Tensor])

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空方阵（空矩阵不具备 LU 语义） | 未覆盖（空输入无定义，见下） |
| 枚举选项 | get_infos 的 True/False 分支 | 已覆盖 |
| 参数类型 | pivot 布尔取值 | 已覆盖 |
| 传参与不传参 | 省略参数与显式传参 | 已覆盖 |
| 等价类/边界值 | 批处理矩阵、pivots 形状 | 已覆盖 |
| 正常传参场景 | 返回 (LU, pivots)，可选 info | 已覆盖 |
| 异常传参场景 | 无主元且 CPU 不支持时报错 | 已覆盖 |
| 混合设备类型 | 单 Tensor 输入，不涉及多设备混合 | 未覆盖（单输入，不适用） |

未覆盖项及原因：
- 空/非空：空矩阵无 LU 因子化定义，未构造该场景。
- 弃用告警：Tensor.lu 的 UserWarning 为进程级 once（同进程仅首次触发一次），
  在用例进程内不可稳定复现，故不对告警单独断言，改由返回结构契约覆盖。
- 混合设备类型：本 API 只涉及单个 Tensor，不存在 NPU/CPU 混合输入场景。
- 职责说明：LU 分解偏计算，用户指定归类为框架类，按框架类模板组织，
  对可在 CPU/NPU 两侧执行的浮点结果以 CPU 为 Golden 对照。
"""
import os
import unittest

import torch

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
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)

if DEVICE_NAME == 'npu':
    import torch_npu.contrib.transfer_to_npu  # noqa: F401


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTensorlu(TestCase):
    """Test cases for torch.Tensor.lu (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_lu_default_returns_two(self):
        """Default call returns (LU, pivots) with matching shapes."""
        lu, pivots = torch.randn(4, 4).lu()
        self.assertEqual(lu.shape, (4, 4))
        self.assertEqual(pivots.shape, (4,))

    def test_lu_get_infos_returns_three(self):
        """get_infos=True adds a 0-dim info tensor."""
        lu, pivots, info = torch.randn(4, 4).lu(get_infos=True)
        self.assertEqual(lu.shape, (4, 4))
        self.assertEqual(info.numel(), 1)

    def test_lu_pivots_dtype(self):
        """The pivots index tensor is an integer tensor."""
        _, pivots = torch.randn(3, 3).lu()
        self.assertTrue(pivots.dtype in (torch.int32, torch.int64))

    def test_lu_batched(self):
        """Batched input keeps the leading batch dimension."""
        lu, pivots = torch.randn(2, 5, 5).lu()
        self.assertEqual(lu.shape, (2, 5, 5))
        self.assertEqual(pivots.shape, (2, 5))

    def test_lu_no_pivot_cpu_raises(self):
        """pivot=False is unsupported on CPU and must raise."""
        x = torch.randn(3, 3)
        with self.assertRaises(RuntimeError):
            x.lu(pivot=False)

    def test_lu_cpu_golden_device(self):
        """Device LU factors match the CPU golden factors."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        torch.manual_seed(0)
        x = torch.randn(4, 4)
        golden_lu, golden_pivots = x.lu()
        dev_lu, dev_pivots = x.to(self.device).lu()
        torch.testing.assert_close(dev_lu.cpu(), golden_lu)
        torch.testing.assert_close(dev_pivots.cpu().to(torch.int64), golden_pivots.to(torch.int64))

    def test_lu_float64_cpu_golden_device(self):
        """float64 device LU factors match the CPU golden factors."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        torch.manual_seed(0)
        x = torch.randn(4, 4, dtype=torch.float64)
        golden_lu, golden_pivots = x.lu()
        dev_lu, dev_pivots = x.to(self.device).lu()
        torch.testing.assert_close(dev_lu.cpu(), golden_lu)
        torch.testing.assert_close(dev_pivots.cpu().to(torch.int64), golden_pivots.to(torch.int64))


if __name__ == "__main__":
    run_tests()
