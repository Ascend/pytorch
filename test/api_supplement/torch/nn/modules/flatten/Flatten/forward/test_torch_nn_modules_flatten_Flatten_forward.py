# -*- coding: utf-8 -*-
"""测试目的：验证 torch.nn.modules.flatten.Flatten.forward 接口功能正确性
API 名称：torch.nn.modules.flatten.Flatten.forward
API 签名：forward(input: Tensor) -> Tensor

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 非空输入为主契约；1 维输入（start_dim=1 超界报 IndexError）边界 | 已覆盖（default_dims/1d_input_raises） |
| 枚举选项 | start_dim/end_dim 取默认、0、中间维、负数多种组合 | 已覆盖（default_dims/full_flatten/middle_dims/single_dim_span） |
| 参数类型 | fp32 输入，输出 dtype 保持；返回值必须为 Tensor | 已覆盖（test_forward_default_dims 同时断言 Tensor 类型与 dtype） |
| 传参与不传参 | 默认 start_dim=1/end_dim=-1 与显式传参两条路径 | 已覆盖（test_forward_default_dims/test_forward_middle_dims） |
| 等价类/边界值 | 全展平 start_dim=0、单维 span no-op、1 维输入超界报错 | 已覆盖（full_flatten/single_dim_span/1d_input_raises） |
| 正常传参场景 | 4 维输入默认展平，NPU/GPU 输出与 CPU Golden 精确对照 | 已覆盖（test_forward_default_dims） |
| 异常传参场景 | start_dim > end_dim 触发 RuntimeError | 已覆盖（test_forward_reversed_dims_raises） |
| 混合设备类型 | CPU 构造模块直接执行设备输入，输出停留在设备侧 | 已覆盖（test_forward_middle_dims） |

未覆盖项及原因：
- 空 Tensor（0 元素）输入：flatten 对空输入行为依赖具体 shape 组合，torch.flatten 对
  0 元素多维输入合法且与设备无关，但 NPU 侧空张量内核覆盖依赖底层算子支持面，
  本文件以 1 维超界报错作为边界等价类代表，空输入未覆盖（底层支持面依赖）。
- 本文件看护 torch.nn.modules.flatten.Flatten.forward（正名路径）；
  模块对象与 extra_repr 契约分别在同名模块对象/extra_repr 用例文件中覆盖。
- torch.nn.Flatten 与 torch.nn.modules.flatten.Flatten 为同一类对象，模块对象文件断言。
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
class TestFlattenForward(TestCase):
    """Cases for torch.nn.modules.flatten.Flatten.forward (NPU/CUDA portable)."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)
        torch.manual_seed(42)
        self.x = torch.randn(2, 3, 4, 5)

    def test_forward_default_dims(self):
        """Default start_dim=1/end_dim=-1 flattens trailing dims to (2, 60)."""
        m = torch.nn.modules.flatten.Flatten()
        expected = torch.flatten(self.x, 1, -1)
        out = m(self.x.to(self.device))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.dtype, self.x.dtype)
        self.assertEqual(tuple(out.shape), (2, 60))
        torch.testing.assert_close(out.cpu(), expected)

    def test_forward_full_flatten(self):
        """start_dim=0/end_dim=-1 flattens everything to a 1-D tensor."""
        m = torch.nn.modules.flatten.Flatten(0, -1)
        expected = torch.flatten(self.x, 0, -1)
        out = m(self.x.to(self.device))
        self.assertEqual(tuple(out.shape), (120,))
        torch.testing.assert_close(out.cpu(), expected)

    def test_forward_middle_dims(self):
        """start_dim=1/end_dim=2 merges only the middle dims to (2, 12, 5).

        The module is constructed on CPU and executes device input, so this
        case also covers the CPU-module/device-input contract.
        """
        m = torch.nn.modules.flatten.Flatten(1, 2)
        expected = torch.flatten(self.x, 1, 2)
        out = m(self.x.to(self.device))
        self.assertEqual(out.device.type, self.device_name)
        self.assertEqual(tuple(out.shape), (2, 12, 5))
        torch.testing.assert_close(out.cpu(), expected)

    def test_forward_single_dim_span(self):
        """start_dim == end_dim is a no-op preserving the input shape."""
        m = torch.nn.modules.flatten.Flatten(2, 2)
        out = m(self.x.to(self.device))
        self.assertEqual(tuple(out.shape), tuple(self.x.shape))
        torch.testing.assert_close(out.cpu(), self.x)

    def test_forward_1d_input_raises(self):
        """1-D input with default dims (start_dim=1) raises IndexError per the
        torch.flatten dim-range contract (host-side validation)."""
        m = torch.nn.modules.flatten.Flatten()
        x1d = torch.randn(6)
        with self.assertRaises(IndexError):
            m(x1d.to(self.device))

    def test_forward_reversed_dims_raises(self):
        """start_dim > end_dim raises RuntimeError per torch.flatten contract."""
        m = torch.nn.modules.flatten.Flatten(2, 1)
        with self.assertRaises(RuntimeError):
            m(self.x.to(self.device))


if __name__ == "__main__":
    run_tests()
