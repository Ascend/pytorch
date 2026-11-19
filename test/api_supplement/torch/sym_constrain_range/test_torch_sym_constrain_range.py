# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.sym_constrain_range 接口功能正确性
API 名称：torch.sym_constrain_range
API 签名：torch.sym_constrain_range(size, min=None, max=None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | size 为必填 | 未覆盖（不适用） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | Number 入参；非 Number 抛 TypeError | 已覆盖 |
| 传参与不传参 | 只传 size / 传 min / 传 min+max | 已覆盖 |
| 等价类/边界值 | min==max 与 min>max 边界 | 已覆盖 |
| 正常传参场景 | 返回 None 且不改变具体数 | 已覆盖 |
| 异常传参场景 | min>max 抛 RuntimeError、非法关键字抛 TypeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收整数/符号整数、不接收 Tensor，混合设备维度不适用。
- 符号约束真正写入 ShapeEnv 的分支需要完整 shape-env/dynamo 上下文，独立构造
  不稳定，故未覆盖；对具体整数输入该调用为无副作用的 None 返回，已断言。
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
class Testsym_constrain_range(TestCase):
    """Test cases for torch.sym_constrain_range (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_none_for_plain_int(self):
        """Constraining a concrete int is a side-effect-free no-op."""
        self.assertIsNone(torch.sym_constrain_range(3))

    def test_accepts_min_and_max(self):
        """min/max keyword arguments are accepted."""
        self.assertIsNone(torch.sym_constrain_range(3, min=1, max=5))

    def test_min_equals_max_boundary(self):
        """min == max is a valid boundary and returns None."""
        self.assertIsNone(torch.sym_constrain_range(3, min=3, max=3))

    def test_min_greater_than_max_raises(self):
        """min > max violates the contract and raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.sym_constrain_range(3, min=10, max=1)

    def test_non_number_raises(self):
        """A non-numeric size raises TypeError."""
        with self.assertRaises(TypeError):
            torch.sym_constrain_range('not-a-number')

    def test_unexpected_keyword_raises(self):
        """An unexpected keyword raises TypeError."""
        with self.assertRaises(TypeError):
            torch.sym_constrain_range(3, foo=1)


if __name__ == "__main__":
    run_tests()
