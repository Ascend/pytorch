# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.profiler.itt.range 接口功能正确性
API 名称：torch.profiler.itt.range
API 签名：torch.profiler.itt.range(msg, *args, **kwargs)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------------------------------|--------------------------------------------------|
| 空/非空 | 空消息字符串与非空消息字符串 | 已覆盖（basic_and_empty_message） |
| 枚举选项 | 枚举参数各合法取值 | 未覆盖，该接口无枚举参数 |
| 参数类型 | 位置参数、关键字参数及混合格式化参数 | 已覆盖（positional_and_keyword_formatting） |
| 传参与不传参 | 省略/传入 *args/**kwargs 及缺 msg | 已覆盖（basic_and_empty_message、positional_and_keyword_formatting） |
| 等价类/边界值 | 上下文管理器、装饰器用法及代码块内抛异常 | 已覆盖（basic_and_empty_message、decorator_usage、exception_still_pops） |
| 正常传参场景 | 正常 push/pop 配对、格式化及装饰器调用 | 已覆盖（basic_and_empty_message、positional_and_keyword_formatting、decorator_usage） |
| 异常传参场景 | 缺少 msg 抛 TypeError 及占位符键缺失 KeyError | 已覆盖（missing_msg_raises、invalid_format_key_raises） |
| 混合设备类型 | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖，该接口为 ITT 范围标记上下文管理器，不涉及设备迁移 |

未覆盖项及原因：
- 枚举选项：该接口无枚举参数。
- 混合设备类型：该接口仅接收格式化消息参数并调用 range_push/range_pop，不涉及 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的非图模式浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest
from unittest.mock import patch

import torch
import torch.profiler.itt as itt

SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"

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
class TestProfilerIttRange(TestCase):
    """Test cases for torch.profiler.itt.range."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_and_empty_message(self):
        """range() pairs range_push and range_pop for normal and empty messages."""
        events = []
        with patch.object(itt, "range_push", side_effect=lambda m: events.append(("push", m))), \
                patch.object(itt, "range_pop", side_effect=lambda: events.append(("pop", None))):
            with itt.range("forward_pass"):
                events.append(("body", None))
            with itt.range(""):
                pass
        expected = [("push", "forward_pass"), ("body", None), ("pop", None), ("push", ""), ("pop", None)]
        self.assertEqual(events, expected)

    def test_positional_and_keyword_formatting(self):
        """range() formats msg using positional, keyword, and mixed arguments."""
        pushed = []
        with patch.object(itt, "range_push", side_effect=pushed.append), \
                patch.object(itt, "range_pop") as mock_pop:
            with itt.range("step_{}_{}", 1, "train"):
                pass
            with itt.range("layer_{name}_{idx}", name="conv", idx=3):
                pass
            with itt.range("{0}_{tag}", "eval", tag="v2"):
                pass
        self.assertEqual(pushed, ["step_1_train", "layer_conv_3", "eval_v2"])
        self.assertEqual(mock_pop.call_count, 3)

    def test_exception_still_pops(self):
        """range_pop is still called when an exception is raised inside the context."""
        events = []
        with patch.object(itt, "range_push", side_effect=lambda m: events.append(("push", m))), \
                patch.object(itt, "range_pop", side_effect=lambda: events.append(("pop", None))):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with itt.range("failing_scope"):
                    events.append(("body", None))
                    raise RuntimeError("boom")
        self.assertEqual(events, [("push", "failing_scope"), ("body", None), ("pop", None)])

    def test_decorator_usage(self):
        """itt.range() works as a function decorator and preserves return values."""
        events = []

        @itt.range("fn_{}", 1)
        def compute(x, y):
            return x + y
        with patch.object(itt, "range_push", side_effect=lambda m: events.append(("push", m))), \
                patch.object(itt, "range_pop", side_effect=lambda: events.append(("pop", None))):
            self.assertEqual(compute(2, 3), 5)
        self.assertEqual(events, [("push", "fn_1"), ("pop", None)])

    def test_missing_msg_raises(self):
        """Calling itt.range() without the required msg argument raises TypeError."""
        with self.assertRaises(TypeError), itt.range():
            pass

    def test_invalid_format_key_raises(self):
        """Unmatched format placeholder raises KeyError before push or pop."""
        with patch.object(itt, "range_push") as mock_push, patch.object(itt, "range_pop") as mock_pop:
            with self.assertRaises(KeyError), itt.range("missing_{unknown_key}"):
                pass
        self.assertEqual(mock_push.call_count, 0)
        self.assertEqual(mock_pop.call_count, 0)


if __name__ == "__main__":
    run_tests()
