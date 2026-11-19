# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.init_num_threads 接口功能正确性
API 名称：torch.init_num_threads
API 签名：torch.init_num_threads() -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无入参 | 未覆盖（无入参） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 重复调用幂等 | 已覆盖 |
| 正常传参场景 | 初始化线程池且返回 None | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 无入参、不接收 Tensor，参数与混合设备维度不适用。
- 异常传参场景：无入参，未找到稳定异常路径。
- 本 API 仅初始化全局线程池、不产生浮点输出，故不做跨设备数值对照。
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
class Testinit_num_threads(TestCase):
    """Test cases for torch.init_num_threads (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_none(self):
        """Initialization returns None."""
        self.assertIsNone(torch.init_num_threads())

    def test_repeated_calls_are_idempotent(self):
        """Calling it again does not error and still returns None."""
        self.assertIsNone(torch.init_num_threads())
        self.assertIsNone(torch.init_num_threads())


if __name__ == "__main__":
    run_tests()
