# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.crow_indices 接口功能正确性
API 名称：torch.Tensor.crow_indices
API 签名：Tensor.crow_indices() -> IntTensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 CSR 与非空 CSR | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 与构造时传入的 crow 完全一致 | 已覆盖 |
| 正常传参场景 | CSR 张量返回压缩行索引 | 已覆盖 |
| 异常传参场景 | 非压缩行布局输入报错 | 已覆盖 |
| 混合设备类型 | 单 Tensor 输入，不涉及多设备混合 | 未覆盖（单输入，不适用） |

未覆盖项及原因：
- 枚举/参数类型/传参：crow_indices 无任何参数，相关维度不适用。
- 混合设备类型：本 API 只涉及单个 Tensor，不存在 NPU/CPU 混合输入场景。
- 职责说明：稀疏布局访问器（偏计算），用户指定归类为框架类，按框架类模板组织，
  对可在 CPU/NPU 两侧执行的结果以 CPU 为 Golden 对照。
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


def build_csr(device='cpu'):
    """Build a deterministic CSR tensor [[1, 0], [0, 2], [0, 0]] on device."""
    crow = torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1], dtype=torch.int32, device=device)
    values = torch.tensor([1.0, 2.0], device=device)
    return torch.sparse_csr_tensor(crow, col, values, size=(3, 2))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTensorcrow_indices(TestCase):
    """Test cases for torch.Tensor.crow_indices (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_crow_indices_returns_expected(self):
        """A CSR tensor returns exactly the crow index it was built with."""
        csr = build_csr()
        self.assertEqual(csr.crow_indices().tolist(), [0, 1, 2, 2])

    def test_crow_indices_empty_csr(self):
        """An empty CSR tensor returns a crow index of the expected length."""
        crow = torch.tensor([0, 0], dtype=torch.int32)
        col = torch.empty(0, dtype=torch.int32)
        csr = torch.sparse_csr_tensor(crow, col, torch.empty(0), size=(1, 0))
        self.assertEqual(csr.crow_indices().numel(), 2)
        self.assertEqual(csr.crow_indices().tolist(), [0, 0])

    def test_crow_indices_wrong_layout_raises(self):
        """A dense (strided) tensor rejects crow_indices with a RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, 'crow_indices expected'):
            torch.randn(3).crow_indices()

    def test_crow_indices_coo_raises(self):
        """A COO tensor also rejects crow_indices."""
        coo = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (3, 3))
        with self.assertRaisesRegex(RuntimeError, 'crow_indices expected'):
            coo.crow_indices()

    def test_crow_indices_cpu_golden_device(self):
        """Device CSR crow index equals the CPU golden result."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        golden = build_csr('cpu').crow_indices()
        actual = build_csr(self.device_name).crow_indices().cpu()
        torch.testing.assert_close(actual, golden)


if __name__ == "__main__":
    run_tests()
