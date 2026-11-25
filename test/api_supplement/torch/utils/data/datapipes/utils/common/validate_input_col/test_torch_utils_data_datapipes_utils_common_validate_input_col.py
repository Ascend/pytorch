# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.data.datapipes.utils.common.validate_input_col 接口功能正确性
API 名称：torch.utils.data.datapipes.utils.common.validate_input_col
API 签名：torch.utils.data.datapipes.utils.common.validate_input_col(fn, input_col)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | input_col=None 与非空索引/序列             | 已覆盖（single_arg_valid、multi_arg_valid）            |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | input_col 为 None、int、tuple、list        | 已覆盖（single_arg_valid、multi_arg_valid）            |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（fn 与 input_col 均为必填位置参数）             |
| 等价类/边界值    | *args/默认参数函数与参数数量不匹配边界     | 已覆盖（varargs_and_defaults、mismatch_raises）        |
| 正常传参场景     | 单参、多参、可变参及默认参数函数校验通过   | 已覆盖（single_arg_valid、multi_arg_valid、varargs_and_defaults） |
| 异常传参场景     | 参数数不匹配、无默认仅关键字参数、缺参     | 已覆盖（mismatch_raises、kw_only_raises、missing_args_raises） |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯 Python 函数签名校验工具，无 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：fn 与 input_col 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯 Python 函数签名校验工具，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.data.datapipes.utils.common import validate_input_col

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
class TestValidateInputCol(TestCase):
    """Test cases for torch.utils.data.datapipes.utils.common.validate_input_col."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_single_arg_valid(self):
        """Single-argument callable validates with input_col=None and input_col=0."""
        fn = lambda x: x + 1  # noqa: E731
        self.assertIsNone(validate_input_col(fn, None))
        self.assertIsNone(validate_input_col(fn, 0))

    def test_multi_arg_valid(self):
        """Two-argument callable validates with tuple and list of length 2."""
        fn = lambda x, y: x + y  # noqa: E731
        self.assertIsNone(validate_input_col(fn, (0, 1)))
        self.assertIsNone(validate_input_col(fn, [0, 1]))

    def test_varargs_and_defaults(self):
        """Var-positional *args and defaulted positional parameters validate properly."""
        fn_var = lambda *args: sum(args)  # noqa: E731
        self.assertIsNone(validate_input_col(fn_var, None))
        self.assertIsNone(validate_input_col(fn_var, [0, 1, 2]))

        def fn_def(a, b=1, *, c=2):
            return a + b + c

        self.assertIsNone(validate_input_col(fn_def, 0))
        self.assertIsNone(validate_input_col(fn_def, [0, 1]))

    def test_mismatch_raises(self):
        """Argument count mismatch with input_col size raises ValueError."""
        fn_one = lambda x: x  # noqa: E731
        with self.assertRaises(ValueError):
            validate_input_col(fn_one, (0, 1))

        fn_two = lambda x, y: x + y  # noqa: E731
        with self.assertRaises(ValueError):
            validate_input_col(fn_two, 0)

    def test_kw_only_raises(self):
        """Non-default keyword-only parameter raises ValueError."""
        def fn_kw(a, *, b):
            return a + b

        with self.assertRaises(ValueError):
            validate_input_col(fn_kw, 0)

    def test_missing_args_raises(self):
        """Calling validate_input_col with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            validate_input_col()
        with self.assertRaises(TypeError):
            validate_input_col(lambda x: x)


if __name__ == "__main__":
    run_tests()
