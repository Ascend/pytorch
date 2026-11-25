# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.flop_counter.shape_wrapper 接口功能正确性
API 名称：torch.utils.flop_counter.shape_wrapper
API 签名：torch.utils.flop_counter.shape_wrapper(f)

覆盖维度表：
| 覆盖维度      | 说明                               | 覆盖情况                                               |
|---------------|------------------------------------|--------------------------------------------------------|
| 空/非空       | out_val=None 与非空 Tensor 包装    | 已覆盖（default_out_val_is_none、tensor_args_and_out_val） |
| 枚举选项      | 枚举型参数选项                     | 未覆盖（接口仅接收可调用对象 f，无枚举选项）           |
| 参数类型      | Tensor、嵌套容器与非 Tensor 透传   | 已覆盖（tensor_args_and_out_val、non_tensor_passthrough_and_nested） |
| 传参与不传参  | out_val 默认不传与显式传参         | 已覆盖（default_out_val_is_none、tensor_args_and_out_val） |
| 等价类/边界值 | 嵌套结构透传与函数元数据保留       | 已覆盖（non_tensor_passthrough_and_nested、preserves_metadata） |
| 正常传参场景  | 装饰器转换 Tensor 为 shape 并执行  | 已覆盖（tensor_args_and_out_val、preserves_metadata）  |
| 异常传参场景  | 缺少必需参数 f 抛出 TypeError      | 已覆盖（missing_arg_raises）                           |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合      | 未覆盖（仅提取 Tensor.shape，不执行跨设备张量计算）    |

未覆盖项及原因：
- 枚举选项：shape_wrapper 仅接收单个可调用对象参数 f，无枚举类参数。
- 混合设备类型：包装器仅将 Tensor 映射为 torch.Size 形状对象，不涉及跨设备张量计算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.flop_counter import shape_wrapper

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

# Probe device only when RUN_FLAG is enabled.
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
class TestShapeWrapper(TestCase):
    """Test cases for torch.utils.flop_counter.shape_wrapper."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name)

    def test_tensor_args_and_out_val(self):
        """Tensor args and out_val are converted to torch.Size on CPU and target device."""
        recorded = {}

        @shape_wrapper
        def dummy_formula(a_shape, b_shape, out_shape=None):
            recorded["a"] = a_shape
            recorded["b"] = b_shape
            recorded["out"] = out_shape
            return a_shape[0] * b_shape[1]

        for dev in (torch.device("cpu"), self.device):
            a = torch.empty(3, 4, device=dev)
            b = torch.empty(4, 5, device=dev)
            out = torch.empty(3, 5, device=dev)
            res = dummy_formula(a, b, out_val=out)
            self.assertEqual(res, 15)
            self.assertEqual(recorded["a"], torch.Size([3, 4]))
            self.assertEqual(recorded["b"], torch.Size([4, 5]))
            self.assertEqual(recorded["out"], torch.Size([3, 5]))

    def test_default_out_val_is_none(self):
        """Omitting out_val passes out_shape=None to the wrapped function."""
        @shape_wrapper
        def capture(x_shape, out_shape=None):
            return (x_shape, out_shape)

        x = torch.empty(2, 6, device=self.device)
        self.assertEqual(capture(x), (torch.Size([2, 6]), None))

    def test_non_tensor_passthrough_and_nested(self):
        """Non-tensor arguments pass through unchanged and nested tensors map to shapes."""
        @shape_wrapper
        def capture(pair, flag, label=None, out_shape=None):
            return (pair, flag, label, out_shape)

        t = torch.empty(2, 3, device=self.device)
        result = capture((t, 42), True, label="ok", out_val=[t, None])
        self.assertEqual(result[0], (torch.Size([2, 3]), 42))
        self.assertIs(result[1], True)
        self.assertEqual(result[2], "ok")
        self.assertEqual(result[3], [torch.Size([2, 3]), None])

    def test_preserves_metadata(self):
        """Wrapped function preserves original __name__ and __doc__ metadata."""
        def sample_fn(a_shape, out_shape=None):
            """Sample docstring."""
            return a_shape

        wrapped = shape_wrapper(sample_fn)
        self.assertEqual(wrapped.__name__, "sample_fn")
        self.assertEqual(wrapped.__doc__, "Sample docstring.")

    def test_missing_arg_raises(self):
        """Calling shape_wrapper without a callable raises TypeError."""
        with self.assertRaises(TypeError):
            shape_wrapper()


if __name__ == "__main__":
    run_tests()
