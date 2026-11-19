# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.set_autocast_gpu_dtype 接口功能正确性
API 名称：torch.set_autocast_gpu_dtype
API 签名：torch.set_autocast_gpu_dtype(dtype) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | dtype 为必填 | 未覆盖（不适用） |
| 枚举选项 | float16 / bfloat16 两个取值 | 已覆盖 |
| 参数类型 | torch.dtype 入参 | 已覆盖 |
| 传参与不传参 | dtype 为必填参数 | 已覆盖 |
| 等价类/边界值 | 设置后经 get_autocast_dtype('cuda') 可读回 | 已覆盖 |
| 正常传参场景 | 设置成功且返回 None | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 只接收 dtype、不接收 Tensor，空值与混合设备维度不适用。
- 异常传参场景：入参为 torch.dtype，非 dtype 由 Python 层拒绝，未约定稳定异常，
  故不构造非法入参断言。
- 本 API 为已弃用别名（等价于 set_autocast_dtype('cuda', dtype)），仅写 GPU
  autocast 配置，无 NPU 执行路径；全局状态在 tearDown 中还原。
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
class Testset_autocast_gpu_dtype(TestCase):
    """Test cases for torch.set_autocast_gpu_dtype (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved = torch.get_autocast_dtype('cuda')

    def tearDown(self):
        try:
            torch.set_autocast_dtype('cuda', self._saved)
        finally:
            super().tearDown()

    def test_returns_none(self):
        """The setter returns None."""
        self.assertIsNone(torch.set_autocast_gpu_dtype(torch.float16))

    def test_float16_roundtrip(self):
        """Setting the GPU dtype to float16 is reflected by the query."""
        torch.set_autocast_gpu_dtype(torch.float16)
        self.assertEqual(torch.get_autocast_dtype('cuda'), torch.float16)

    def test_bfloat16_roundtrip(self):
        """Setting the GPU dtype to bfloat16 is reflected by the query."""
        torch.set_autocast_gpu_dtype(torch.bfloat16)
        self.assertEqual(torch.get_autocast_dtype('cuda'), torch.bfloat16)


if __name__ == "__main__":
    run_tests()
