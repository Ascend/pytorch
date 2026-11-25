# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.model_dump.get_inline_skeleton 接口功能正确性
API 名称：torch.utils.model_dump.get_inline_skeleton
API 签名：torch.utils.model_dump.get_inline_skeleton()

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 无参接口返回非空 HTML 骨架字符串     | 已覆盖（returns_html_skeleton）                    |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无参数接口，无枚举选项）                   |
| 参数类型         | 输入参数类型校验                     | 未覆盖（无参数接口，非法传参见异常场景）           |
| 传参与不传参     | 无参正常调用与多余传参异常           | 已覆盖（returns_html_skeleton、                    |
|                  |                                      | unexpected_arg_raises）                            |
| 等价类/边界值    | 内联 JS 资产、占位符与多次调用确定性 | 已覆盖（inlines_js_assets、                        |
|                  |                                      | contains_burn_in_placeholder、                     |
|                  |                                      | deterministic_across_calls）                       |
| 正常传参场景     | 返回包含内联脚本与占位符的 HTML      | 已覆盖（returns_html_skeleton、inlines_js_assets、 |
|                  |                                      | contains_burn_in_placeholder）                     |
| 异常传参场景     | 传入多余位置或关键字参数抛 TypeError | 已覆盖（unexpected_arg_raises）                    |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯静态 HTML 模板生成，无 Tensor 输入）     |

未覆盖项及原因：
- 枚举选项：该接口为无参函数，不存在枚举类参数。
- 参数类型：该接口不接受任何参数，传参校验已在异常传参场景覆盖。
- 混合设备类型：该接口仅读取并内联内置 HTML/JS 静态资源，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.model_dump import get_inline_skeleton

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
class TestGetInlineSkeleton(TestCase):
    """Test cases for torch.utils.model_dump.get_inline_skeleton."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_returns_html_skeleton(self):
        """get_inline_skeleton returns a non-empty HTML document string."""
        html = get_inline_skeleton()
        self.assertIsInstance(html, str)
        self.assertGreater(len(html), 0)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html>", html)
        self.assertIn("</html>", html)
        self.assertIn("<script", html)

    def test_inlines_js_assets(self):
        """External code.js and unpkg module URLs are inlined into the skeleton."""
        html = get_inline_skeleton()
        self.assertNotIn('src="./code.js"', html)
        self.assertNotIn("https://unpkg.com/preact?module", html)
        self.assertNotIn("https://unpkg.com/htm?module", html)
        self.assertIn("data:application/javascript,", html)

    def test_contains_burn_in_placeholder(self):
        """Returned skeleton includes the BURNED_IN_MODEL_INFO placeholder."""
        html = get_inline_skeleton()
        self.assertIn("BURNED_IN_MODEL_INFO = null", html)

    def test_deterministic_across_calls(self):
        """Repeated calls to get_inline_skeleton return identical HTML strings."""
        first = get_inline_skeleton()
        second = get_inline_skeleton()
        self.assertEqual(first, second)

    def test_unexpected_arg_raises(self):
        """Passing positional or keyword arguments raises TypeError."""
        with self.assertRaises(TypeError):
            get_inline_skeleton("extra")
        with self.assertRaises(TypeError):
            get_inline_skeleton(foo=1)


if __name__ == "__main__":
    run_tests()
