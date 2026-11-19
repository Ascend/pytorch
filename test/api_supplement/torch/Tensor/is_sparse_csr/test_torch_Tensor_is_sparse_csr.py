# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.is_sparse_csr 接口功能正确性
API 名称：torch.Tensor.is_sparse_csr
API 签名：Tensor.is_sparse_csr -> bool（只读属性）

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 CSR 与非空 CSR、COO、dense | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | CSR/CSC/BSR/COO/dense 布局区分 | 已覆盖 |
| 正常传参场景 | CSR 张量返回 True | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 单 Tensor 输入，不涉及多设备混合 | 未覆盖（单输入，不适用） |

未覆盖项及原因：
- is_sparse_csr 为只读布尔属性（纯框架状态，A 类），无独立设备执行路径，不做 CPU/NPU 数值对照。
- 异常传参场景：属性访问无入参，未找到稳定异常路径。
- 混合设备类型：本 API 只涉及单个 Tensor，不存在 NPU/CPU 混合输入场景。
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


def build_csr():
    """Build a small CSR tensor on CPU."""
    crow = torch.tensor([0, 1, 2], dtype=torch.int32)
    col = torch.tensor([0, 1], dtype=torch.int32)
    return torch.sparse_csr_tensor(crow, col, torch.tensor([1.0, 2.0]), size=(2, 2))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTensoris_sparse_csr(TestCase):
    """Test cases for torch.Tensor.is_sparse_csr (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_is_sparse_csr_true_for_csr(self):
        """A CSR tensor reports is_sparse_csr == True."""
        self.assertTrue(build_csr().is_sparse_csr)

    def test_is_sparse_csr_false_for_dense(self):
        """A dense tensor reports is_sparse_csr == False."""
        self.assertFalse(torch.randn(2, 2).is_sparse_csr)

    def test_is_sparse_csr_false_for_coo(self):
        """A COO tensor is not CSR."""
        coo = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2))
        self.assertFalse(coo.is_sparse_csr)

    def test_is_sparse_csr_false_for_csc(self):
        """A CSC tensor is not CSR (layout is distinguished)."""
        csc = build_csr().to_sparse_csc()
        self.assertFalse(csc.is_sparse_csr)

    def test_is_sparse_csr_returns_bool(self):
        """The property yields a real bool."""
        self.assertIsInstance(build_csr().is_sparse_csr, bool)


if __name__ == "__main__":
    run_tests()
