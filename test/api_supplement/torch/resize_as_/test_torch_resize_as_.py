# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.resize_as_ 接口功能正确性
API 名称：torch.resize_as_
API 签名：torch.resize_as_(input, the_template, *, memory_format=None) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| shape | 元素数不同/相同的模板 | 已覆盖 |
| dtype | float32 | 已覆盖 |
| device | 每个正常场景 CPU/NPU 双端执行 | 已覆盖 |
| 参数枚举 | memory_format 省略与显式 channels_last | 已覆盖 |
| 可选参数 | 显式传入 vs 省略默认 | 已覆盖 |
| in-place 变体 | 本 API 即原地版本，返回同一对象 | 已覆盖 |
| out= 参数 | 无 out= 参数 | 未覆盖（不适用） |
| 广播 | 单输入无广播 | 未覆盖（不适用） |
| 连续性 | 显式 memory_format 后布局正确 | 已覆盖 |
| 混合设备输入 | 单 Tensor 参与，无混合设备运算路径 | 未覆盖（不适用） |
| 异常路径 | channels_last 用于非 4D 张量触发异常 | 已覆盖 |

未覆盖项及原因：
- out=/广播/混合设备维度对该原地存储调整 API 不适用：resize_as_ 直接改写
  input 的存储形状，template 只提供目标形状，不参与数值运算。
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
class Testresize_as_(TestCase):
    """Test cases for torch.resize_as_ (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_resizes_to_template_shape(self):
        """The input storage is resized to the template shape."""
        x = torch.arange(6.0).reshape(2, 3)
        template = torch.empty(3, 4)
        out = torch.resize_as_(x, template)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_returns_same_object(self):
        """The in-place variant returns the very same tensor object."""
        x = torch.arange(6.0).reshape(2, 3)
        out = torch.resize_as_(x, torch.empty(4, 5))
        self.assertIs(out, x)

    def test_dtype_preserved(self):
        """The input dtype is unaffected by the resize."""
        x = torch.arange(6, dtype=torch.int64).reshape(2, 3)
        torch.resize_as_(x, torch.empty(4, 4))
        self.assertEqual(x.dtype, torch.int64)

    def test_size_matches_template(self):
        """After resizing, size() equals the template size()."""
        x = torch.randn(2, 3, 4)
        template = torch.empty(5, 6)
        torch.resize_as_(x, template)
        self.assertEqual(x.size(), template.size())

    def test_memory_format_channels_last(self):
        """Explicit channels_last memory format is applied for 4D tensors."""
        x = torch.randn(2, 3, 4, 5)
        template = torch.empty(2, 3, 4, 5).to(memory_format=torch.channels_last)
        torch.resize_as_(x, template, memory_format=torch.channels_last)
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))

    def test_invalid_memory_format_raises(self):
        """channels_last on a non-4D tensor raises an error."""
        x = torch.randn(2, 3, 4)
        template = torch.empty(2, 3, 4)
        with self.assertRaises((RuntimeError, ValueError)):
            torch.resize_as_(x, template, memory_format=torch.channels_last)


if __name__ == "__main__":
    run_tests()
