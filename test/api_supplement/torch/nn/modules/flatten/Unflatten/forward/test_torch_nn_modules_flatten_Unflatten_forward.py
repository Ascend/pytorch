# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Unflatten.forward 接口功能正确性
API 名称：torch.nn.modules.flatten.Unflatten.forward
API 签名：forward(input: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空输入为主契约；dim=0（1 维输入）边界 | 已覆盖（test_forward_default_dims/test_forward_dim0） |
| 枚举选项 | dim 取正数、0、负数；容器取 tuple/list/torch.Size | 已覆盖（default_dims（tuple/torch.Size）/list_size/dim0/negative_dim） |
| 参数类型 | fp32 输入，输出 dtype 保持；返回值必须为 Tensor | 已覆盖（test_forward_default_dims） |
| 传参与不传参 | forward 仅 input 参数；dim/unflattened_size 来自构造 | 已覆盖（test_forward_default_dims） |
| 等价类/边界值 | 与 Flatten 组合往返恢复原形状与数值 | 已覆盖（test_forward_round_trip_with_flatten） |
| 正常传参场景 | 2 维输入 dim=1 展开，NPU/GPU 输出与 CPU Golden 精确对照 | 已覆盖（test_forward_default_dims） |
| 异常传参场景 | unflattened_size 乘积不等于 dim 大小触发 RuntimeError | 已覆盖（test_forward_size_mismatch_raises） |
| 混合设备类型 | CPU 构造模块直接执行设备输入，输出停留在设备侧 | 已覆盖（test_forward_default_dims） |

未覆盖项及原因：
- 空 Tensor（0 元素）输入：unflatten 对空输入行为依赖具体 shape 组合，NPU 侧空张量内核
  覆盖依赖底层算子支持面，本文件未覆盖（底层支持面依赖）。
- 本文件看护 torch.nn.modules.flatten.Unflatten.forward（正名路径）；
  模块对象与 extra_repr 契约分别在同名模块对象/extra_repr 用例文件中覆盖。
- torch.nn.Unflatten 与 torch.nn.modules.flatten.Unflatten 为同一类对象，模块对象文件断言。
- 全部正常用例以同输入 CPU 结果为 Golden，使用 torch.testing.assert_close 对照。

注意：forward 输出为浮点 Tensor，CPU/GPU 或 CPU/NPU 对照统一使用
     torch.testing.assert_close（设备无关写法，两侧一致）；
     形状、dtype 与异常路径使用精确断言。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"

DEVICE_NAME = "cpu"
if RUN_FLAG:
    DEVICE_NAME = None
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    except (ImportError, AttributeError):
        pass
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestUnflattenForward(TestCase):
    """Cases for torch.nn.modules.flatten.Unflatten.forward (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)
        torch.manual_seed(42)
        self.x = torch.randn(2, 50)

    def test_forward_default_dims(self):
        """dim=1 expands (2, 50) to (2, 2, 5, 5) for tuple and torch.Size sizes.

        The module is constructed on CPU and executes device input, so this
        case also covers the CPU-module/device-input contract.
        """
        expected = torch.unflatten(self.x, 1, (2, 5, 5))
        for size in ((2, 5, 5), torch.Size([2, 5, 5])):
            m = torch.nn.modules.flatten.Unflatten(1, size)
            out = m(self.x.to(self.device))
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, self.x.dtype)
            self.assertEqual(out.device.type, self.device_name)
            self.assertEqual(tuple(out.shape), (2, 2, 5, 5))
            torch.testing.assert_close(out.cpu(), expected)

    def test_forward_dim0(self):
        """dim=0 on a 1-D input expands (50,) to (2, 5, 5)."""
        m = torch.nn.modules.flatten.Unflatten(0, (2, 5, 5))
        x1d = torch.randn(50)
        out = m(x1d.to(self.device))
        self.assertEqual(tuple(out.shape), (2, 5, 5))
        torch.testing.assert_close(out.cpu(), torch.unflatten(x1d, 0, (2, 5, 5)))

    def test_forward_negative_dim(self):
        """dim=-1 expands the last dim of (2, 50) to (2, 2, 5, 5)."""
        m = torch.nn.modules.flatten.Unflatten(-1, (2, 5, 5))
        out = m(self.x.to(self.device))
        self.assertEqual(tuple(out.shape), (2, 2, 5, 5))
        torch.testing.assert_close(
            out.cpu(), torch.unflatten(self.x, -1, (2, 5, 5))
        )

    def test_forward_list_size(self):
        """List unflattened_size path matches the tuple path."""
        m = torch.nn.modules.flatten.Unflatten(1, [10, 5])
        out = m(self.x.to(self.device))
        self.assertEqual(tuple(out.shape), (2, 10, 5))
        torch.testing.assert_close(
            out.cpu(), torch.unflatten(self.x, 1, (10, 5))
        )

    def test_forward_round_trip_with_flatten(self):
        """Flatten then Unflatten restores the original 4-D shape and values."""
        flatten_m = torch.nn.modules.flatten.Flatten(1)
        unflatten_m = torch.nn.modules.flatten.Unflatten(1, (3, 4, 5))
        x = torch.randn(2, 3, 4, 5)
        flattened = flatten_m(x.to(self.device))
        self.assertEqual(tuple(flattened.shape), (2, 60))
        restored = unflatten_m(flattened)
        self.assertEqual(tuple(restored.shape), (2, 3, 4, 5))
        torch.testing.assert_close(restored.cpu(), x)

    def test_forward_size_mismatch_raises(self):
        """A product mismatch between unflattened_size and dim raises RuntimeError."""
        m = torch.nn.modules.flatten.Unflatten(1, (2, 5, 5))
        with self.assertRaises(RuntimeError):
            m(torch.randn(2, 10).to(self.device))


if __name__ == "__main__":
    run_tests()
