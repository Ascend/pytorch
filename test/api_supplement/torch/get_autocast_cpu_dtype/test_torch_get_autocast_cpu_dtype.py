# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.get_autocast_cpu_dtype 接口功能正确性
API 名称：torch.get_autocast_cpu_dtype
API 签名：torch.get_autocast_cpu_dtype() -> torch.dtype

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 无入参 | 未覆盖（无入参） |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 无入参 | 未覆盖（无入参） |
| 传参与不传参 | 无参数 | 未覆盖（无参数） |
| 等价类/边界值 | 默认值 bfloat16、set 后 roundtrip | 已覆盖 |
| 正常传参场景 | 返回 CPU autocast 的 dtype | 已覆盖 |
| 异常传参场景 | 无稳定异常路径 | 未覆盖（见下） |
| 混合设备类型 | 无 Tensor 输入 | 未覆盖（不适用） |

未覆盖项及原因：
- 本 API 无入参、不接收 Tensor，参数与混合设备维度不适用。
- 异常传参场景：无入参，未找到稳定异常路径。
- 本 API 面向 CPU autocast 配置（B 类硬件感知工具），NPU/CUDA 路径由 set_autocast_gpu_dtype
  与 set_autocast_enabled 覆盖，故本文件不做跨设备数值对照。
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
class Testget_autocast_cpu_dtype(TestCase):
    """Test cases for torch.get_autocast_cpu_dtype (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._saved_dtype = torch.get_autocast_dtype('cpu')

    def tearDown(self):
        try:
            torch.set_autocast_dtype('cpu', self._saved_dtype)
        finally:
            super().tearDown()

    def test_default_is_bfloat16(self):
        """The default CPU autocast dtype is bfloat16."""
        self.assertEqual(torch.get_autocast_cpu_dtype(), torch.bfloat16)

    def test_returns_torch_dtype(self):
        """The getter returns a torch.dtype instance."""
        self.assertIsInstance(torch.get_autocast_cpu_dtype(), torch.dtype)

    def test_roundtrip_after_set(self):
        """After changing the CPU dtype the getter reflects the new value."""
        torch.set_autocast_cpu_dtype(torch.float16)
        self.assertEqual(torch.get_autocast_cpu_dtype(), torch.float16)
        torch.set_autocast_cpu_dtype(torch.bfloat16)
        self.assertEqual(torch.get_autocast_cpu_dtype(), torch.bfloat16)


if __name__ == "__main__":
    run_tests()
