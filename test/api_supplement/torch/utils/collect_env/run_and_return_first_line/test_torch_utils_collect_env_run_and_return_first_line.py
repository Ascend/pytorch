# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.run_and_return_first_line 接口功能正确性
API 名称：torch.utils.collect_env.run_and_return_first_line
API 签名：torch.utils.collect_env.run_and_return_first_line(run_lambda, command)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空 stdout 与非空 stdout 返回行为           | 已覆盖（empty_stdout、single_line_stdout）             |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 可调用对象 run_lambda 与字符串命令         | 已覆盖（multiline_stdout、single_line_stdout）         |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（run_lambda/command 均为必填，无可选参数）      |
| 等价类/边界值    | 多行截取首行、单行、首字符换行、非零退出码 | 已覆盖（leading_newline、nonzero_rc_returns_none）     |
| 正常传参场景     | rc == 0 返回首行文本                       | 已覆盖（multiline_stdout、single_line_stdout）         |
| 异常传参场景     | 缺失参数或 run_lambda 不可调用             | 已覆盖（missing_arg_and_non_callable_raises）          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯主机命令输出解析函数，不接收 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：run_lambda 与 command 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯主机命令输出首行提取辅助函数，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.collect_env import run_and_return_first_line

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
class TestRunAndReturnFirstLine(TestCase):
    """Test cases for torch.utils.collect_env.run_and_return_first_line."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_multiline_stdout(self):
        """Returns only the first line from multiline stdout and forwards command."""
        seen = []

        def fake_run(cmd):
            seen.append(cmd)
            return (0, "first line\nsecond line\nthird line", "")

        res = run_and_return_first_line(fake_run, "uname -a")
        self.assertEqual(res, "first line")
        self.assertEqual(seen, ["uname -a"])

    def test_single_line_stdout(self):
        """Returns the single line unchanged when no newline is present."""
        res = run_and_return_first_line(lambda c: (0, "only line", ""), "cmd")
        self.assertEqual(res, "only line")

    def test_empty_stdout(self):
        """Returns empty string '' when rc == 0 and stdout is empty."""
        res = run_and_return_first_line(lambda c: (0, "", ""), "cmd")
        self.assertEqual(res, "")

    def test_leading_newline(self):
        """Returns '' when stdout starts with a newline character."""
        res = run_and_return_first_line(lambda c: (0, "\nsecond", ""), "cmd")
        self.assertEqual(res, "")

    def test_nonzero_rc_returns_none(self):
        """Returns None when run_lambda returns a non-zero exit code."""
        self.assertIsNone(run_and_return_first_line(lambda c: (1, "first\nsecond", "err"), "cmd"))
        self.assertIsNone(run_and_return_first_line(lambda c: (127, "", "not found"), "cmd"))

    def test_missing_arg_and_non_callable_raises(self):
        """Missing args or non-callable run_lambda raises TypeError."""
        with self.assertRaises(TypeError):
            run_and_return_first_line(lambda c: (0, "a", ""))
        with self.assertRaises(TypeError):
            run_and_return_first_line(None, "cmd")


if __name__ == "__main__":
    run_tests()
