# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.values 接口功能正确性
API 名称：torch.Tensor.values
API 签名：Tensor.values() -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 COO 与非空 COO | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 与构造时传入的 values 完全一致 | 已覆盖 |
| 正常传参场景 | COO 张量返回 values 张量 | 已覆盖 |
| 异常传参场景 | 非 COO 布局输入报错 | 已覆盖 |
| 混合设备类型 | 单 Tensor 输入，不涉及多设备混合 | 未覆盖（单输入，不适用） |

未覆盖项及原因：
- 枚举/参数类型/传参：values 无任何参数，相关维度不适用。
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


def build_coo(device='cpu'):
    """Build a deterministic coalesced COO tensor on device."""
    idx = torch.tensor([[0, 1], [1, 2]], device=device)
    values = torch.tensor([3.0, 4.0], device=device)
    return torch.sparse_coo_tensor(idx, values, (3, 3)).coalesce()


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestTensorvalues(TestCase):
    """Test cases for torch.Tensor.values (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_values_returns_expected(self):
        """A coalesced COO tensor returns exactly its values."""
        coo = build_coo()
        self.assertEqual(coo.values().tolist(), [3.0, 4.0])

    def test_values_empty_coo(self):
        """An empty COO tensor returns an empty values tensor."""
        coo = torch.sparse_coo_tensor(torch.empty(2, 0, dtype=torch.long), torch.empty(0), (3, 3))
        self.assertEqual(coo.values().numel(), 0)

    def test_values_wrong_layout_raises(self):
        """A dense (strided) tensor rejects values with a RuntimeError."""
        with self.assertRaisesRegex(RuntimeError, 'values expected'):
            torch.randn(3).values()

    def test_values_dtype_preserved(self):
        """The returned values keep the original dtype."""
        idx = torch.tensor([[0, 1], [1, 2]])
        coo = torch.sparse_coo_tensor(idx, torch.tensor([1, 2], dtype=torch.int64), (3, 3)).coalesce()
        self.assertEqual(coo.values().dtype, torch.int64)

    def test_values_cpu_golden_device(self):
        """Device COO values equal the CPU golden result."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        golden = build_coo('cpu').values()
        actual = build_coo(self.device_name).values().cpu()
        torch.testing.assert_close(actual, golden)


if __name__ == "__main__":
    run_tests()
