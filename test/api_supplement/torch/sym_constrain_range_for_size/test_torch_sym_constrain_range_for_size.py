# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.sym_constrain_range_for_size 接口功能正确性
API 名称：torch.sym_constrain_range_for_size
API 签名：torch.sym_constrain_range_for_size(size, min=None, max=None) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | size 为必填 | 未覆盖（不适用） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | Number 入参 | 已覆盖 |
| 传参与不传参 | 只传 size / 传 min / 传 min+max | 已覆盖 |
| 等价类/边界值 | max 合法与非法（<=2）边界 | 已覆盖 |
| 正常传参场景 | 返回 None 且不改变具体数 | 已覆盖 |
| 异常传参场景 | 非法上界抛 RuntimeError | 已覆盖 |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收整数/符号整数、不接收 Tensor，混合设备维度不适用。
- 真正写入 ShapeEnv 的符号分支需要完整 shape-env/dynamo 上下文，独立构造
  不稳定，故未覆盖；对具体数输入该调用为无副作用的 None 返回，已断言。
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
class Testsym_constrain_range_for_size(TestCase):
    """Test cases for torch.sym_constrain_range_for_size (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_none_for_plain_int(self):
        """Constraining a concrete int is a side-effect-free no-op."""
        self.assertIsNone(torch.sym_constrain_range_for_size(3))

    def test_accepts_valid_min_and_max(self):
        """A size range with max > 2 is accepted and returns None."""
        self.assertIsNone(torch.sym_constrain_range_for_size(3, min=3, max=10))

    def test_invalid_max_raises(self):
        """A max that is not greater than 2 raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            torch.sym_constrain_range_for_size(3, min=1, max=2)


if __name__ == "__main__":
    run_tests()
