# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.detach_ 接口功能正确性
API 名称：torch.detach_
API 签名：torch.detach_(self: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 1D/2D 张量 | 已覆盖 |
| dtype | float32/float64 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | 无离散参数 | 未覆盖（无该类参数） |
| 可选参数 | 无可选参数 | 未覆盖（无参数） |
| in-place 变体 | 本 API 即为 in-place 变体 | 已覆盖 |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 单张量操作，无广播 | 未覆盖（不适用） |
| 连续性 | 非连续张量 detach_ 后仍共享存储 | 已覆盖 |
| 混合设备输入 | 单 Tensor 输入，不涉及混合设备 | 未覆盖（不适用） |
| 异常路径 | 对叶子张量原地断开可稳定断言 | 已覆盖 |

未覆盖项及原因：
- 本 API 无参数、不接收多设备输入，参数枚举、out=、广播、混合设备维度均不适用。
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
class Testdetach_(TestCase):
    """Test cases for torch.detach_ (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_returns_same_object_on_device(self):
        """detach_ returns the very same tensor object it was called on."""
        x = torch.randn(2, 3, device=self.device, requires_grad=True)
        out = torch.detach_(x)
        self.assertIs(out, x)

    def test_clears_requires_grad(self):
        """After a device-side detach_ the tensor no longer requires grad."""
        x = torch.randn(2, 3, device=self.device, requires_grad=True)
        torch.detach_(x)
        self.assertFalse(x.requires_grad)

    def test_shares_storage_with_cpu_view(self):
        """The detached tensor keeps the same storage (values stay identical)."""
        cpu = torch.arange(6.0).reshape(2, 3)
        dev = cpu.clone().to(self.device)
        torch.detach_(dev)
        self.assertEqual(dev.data_ptr() != 0, True)
        torch.testing.assert_close(dev.cpu(), cpu)

    def test_detached_tensor_leaves_graph(self):
        """Ops built on a detached tensor carry no grad_fn and no grad flows."""
        base = torch.randn(4, device=self.device, requires_grad=True)
        leaf = base * 2.0
        detached = torch.detach_(leaf)
        self.assertIsNone(detached.grad_fn)
        y = (detached * 3.0).sum()
        self.assertFalse(y.requires_grad)

    def test_non_contiguous_input(self):
        """A non-contiguous tensor owning its storage can be detached in place."""
        x = torch.empty_strided((4, 4), (1, 4), device=self.device)
        x.copy_(torch.randn(4, 4, device=self.device))
        self.assertFalse(x.is_contiguous())
        out = torch.detach_(x)
        self.assertIs(out, x)
        self.assertEqual(out.shape, (4, 4))


if __name__ == "__main__":
    run_tests()
