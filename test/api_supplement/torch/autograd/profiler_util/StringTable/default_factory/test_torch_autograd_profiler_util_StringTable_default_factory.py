# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.profiler_util.StringTable.default_factory 属性功能正确性
API 名称：torch.autograd.profiler_util.StringTable.default_factory
API 签名：StringTable.default_factory: Callable | None

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 属性为 None 与可调用对象状态 | 已覆盖（default_is_none、set_callable_and_reset_none） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该属性无枚举选项 |
| 参数类型 | 内置类型、自定义函数及非法对象 | 已覆盖（init_with_factory、set_callable_and_reset_none 等） |
| 传参与不传参 | 构造时省略或传入 factory | 已覆盖（default_is_none、init_with_factory） |
| 等价类/边界值 | 置回 None 及与 __missing__ demangle 交互 | 已覆盖（set_callable_and_reset_none、missing_still_uses_demangle） |
| 正常传参场景 | 读取、初始化与动态修改属性 | 已覆盖（init_with_factory、set_callable_and_reset_none、missing_still_uses_demangle） |
| 异常传参场景 | 非可调用对象触发 TypeError | 已覆盖（non_callable_raises_and_preserves_previous） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该属性为纯 Python defaultdict 属性，不涉及设备迁移 |

未覆盖项及原因：
- 枚举选项：该属性不涉及枚举取值。
- 混合设备类型：该属性仅管理工厂函数引用，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.autograd.profiler_util import StringTable

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
class TestStringTableDefaultFactory(TestCase):
    """Test cases for torch.autograd.profiler_util.StringTable.default_factory."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_default_is_none(self):
        """A freshly constructed StringTable() has default_factory set to None."""
        table = StringTable()
        self.assertIsNone(table.default_factory)

    def test_init_with_factory(self):
        """Passing a factory callable to StringTable(factory) initializes default_factory."""
        table = StringTable(int)
        self.assertIs(table.default_factory, int)

    def test_set_callable_and_reset_none(self):
        """default_factory can be updated to a callable and reset back to None."""
        table = StringTable()

        def factory():
            return "fallback"

        table.default_factory = str
        self.assertIs(table.default_factory, str)

        table.default_factory = factory
        self.assertIs(table.default_factory, factory)

        table.default_factory = None
        self.assertIsNone(table.default_factory)

    def test_missing_still_uses_demangle(self):
        """Setting default_factory does not bypass StringTable.__missing__ demangling."""
        calls = []

        def custom_factory():
            calls.append(1)
            return "from_factory"

        table = StringTable(custom_factory)
        self.assertIs(table.default_factory, custom_factory)

        # StringTable overrides __missing__ directly without calling default_factory.
        short_val = table["t"]
        long_val = table["aten::relu"]
        self.assertEqual(short_val, "t")
        self.assertEqual(long_val, torch._C._demangle("aten::relu"))
        self.assertEqual(len(calls), 0)

    def test_non_callable_raises_and_preserves_previous(self):
        """Passing a non-callable factory to StringTable raises TypeError."""
        with self.assertRaises(TypeError):
            StringTable(123)
        with self.assertRaises(TypeError):
            StringTable("not_callable")
        table = StringTable(str)
        self.assertIs(table.default_factory, str)


if __name__ == "__main__":
    run_tests()
