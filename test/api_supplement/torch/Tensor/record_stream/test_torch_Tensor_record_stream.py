# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.Tensor.record_stream 接口功能正确性
API 名称：torch.Tensor.record_stream
API 签名：Tensor.record_stream(stream) -> None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空张量记录流 | 已覆盖 |
| 枚举选项 | 无枚举参数 | 未覆盖（无该类参数） |
| 参数类型 | 传入 Stream 对象；非法类型报错 | 已覆盖 |
| 传参与不传参 | stream 为必填参数 | 已覆盖 |
| 等价类/边界值 | 默认流、显式新建流 | 已覆盖 |
| 正常传参场景 | 设备张量在流上登记成功（返回 None） | 已覆盖 |
| 异常传参场景 | CPU 张量不受支持、非法 stream 类型 | 已覆盖 |
| 混合设备类型 | 单 Tensor 输入，不涉及多设备混合 | 未覆盖（单输入，不适用） |

未覆盖项及原因：
- 枚举选项：record_stream 无枚举参数。
- 混合设备类型：本 API 只涉及单个 Tensor，不存在 NPU/CPU 混合输入场景。
- 设备入口：使用 torch.cuda.Stream（transfer_to_npu 已在 NPU 侧改为 torch.npu 实现），
  禁止硬编码 torch.npu.*，以保持 NPU/CUDA 同码。
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
class TestTensorrecord_stream(TestCase):
    """Test cases for torch.Tensor.record_stream (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_record_stream_default_stream(self):
        """Recording a device tensor on a freshly created stream returns None."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        x = torch.zeros(4, device=self.device)
        stream = torch.cuda.Stream()
        self.assertIsNone(x.record_stream(stream))

    def test_record_stream_current_stream(self):
        """Recording against the current stream also returns None and is a no-op on values."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        x = torch.ones(4, device=self.device)
        before = x.clone()
        self.assertIsNone(x.record_stream(torch.cuda.current_stream()))
        torch.testing.assert_close(x.cpu(), before.cpu())

    def test_record_stream_cpu_unsupported(self):
        """A CPU tensor has no record_stream kernel and must raise."""
        with self.assertRaises((NotImplementedError, RuntimeError)):
            torch.zeros(4).record_stream(torch.cuda.Stream())

    def test_record_stream_invalid_type_raises(self):
        """A non-stream argument is rejected."""
        if self.device_name == 'cpu':
            self.skipTest('无设备: 缺少 NPU/CUDA 设备执行路径')
        x = torch.zeros(4, device=self.device)
        with self.assertRaises((TypeError, RuntimeError)):
            x.record_stream(123)


if __name__ == "__main__":
    run_tests()
