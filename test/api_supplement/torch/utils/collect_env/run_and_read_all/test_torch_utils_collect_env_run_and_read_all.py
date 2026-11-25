# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.run_and_read_all 接口功能正确性
API 名称：torch.utils.collect_env.run_and_read_all
API 签名：torch.utils.collect_env.run_and_read_all(run_lambda, command)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空 stdout 与非空 stdout 返回行为           | 已覆盖（empty_stdout、basic_success）                  |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 字符串命令与列表形式命令                   | 已覆盖（basic_success、list_command）                  |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（run_lambda/command 均为必填，无可选参数）      |
| 等价类/边界值    | 多行输出、含 stderr、正负非零返回码        | 已覆盖（multiline_stdout、ignores_stderr_on_success、nonzero_rc_returns_none） |
| 正常传参场景     | rc == 0 完整返回 stdout                    | 已覆盖（basic_success、multiline_stdout、list_command） |
| 异常传参场景     | 缺失参数或 run_lambda 不可调用             | 已覆盖（missing_arg_and_non_callable_raises）          |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯主机命令执行辅助函数，不接收 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：run_lambda 与 command 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯主机命令输出读取辅助函数，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.collect_env import run_and_read_all

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
class TestRunAndReadAll(TestCase):
    """Test cases for torch.utils.collect_env.run_and_read_all."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_success(self):
        """Returns stdout string when rc == 0 and forwards command unchanged."""
        seen = []

        def fake_run(cmd):
            seen.append(cmd)
            return (0, "hello world", "")

        res = run_and_read_all(fake_run, "echo hello")
        self.assertEqual(res, "hello world")
        self.assertEqual(seen, ["echo hello"])

    def test_multiline_stdout(self):
        """Preserves full multiline stdout without truncation."""
        text = "line1\nline2\nline3"
        res = run_and_read_all(lambda c: (0, text, ""), "cat file")
        self.assertEqual(res, text)

    def test_empty_stdout(self):
        """Returns empty string '' (not None) when rc == 0 and stdout is empty."""
        res = run_and_read_all(lambda c: (0, "", ""), "true")
        self.assertEqual(res, "")

    def test_ignores_stderr_on_success(self):
        """Non-empty stderr does not prevent returning stdout when rc == 0."""
        res = run_and_read_all(lambda c: (0, "output", "warning"), "cmd")
        self.assertEqual(res, "output")

    def test_list_command(self):
        """Accepts list-form command and passes it directly to run_lambda."""
        seen = []
        res = run_and_read_all(lambda c: (seen.append(c) or 0, "ok", ""), ["echo", "ok"])
        self.assertEqual(res, "ok")
        self.assertEqual(seen, [["echo", "ok"]])

    def test_nonzero_rc_returns_none(self):
        """Returns None for any non-zero exit code."""
        self.assertIsNone(run_and_read_all(lambda c: (1, "partial", "err"), "cmd"))
        self.assertIsNone(run_and_read_all(lambda c: (-1, "partial", "err"), "cmd"))

    def test_missing_arg_and_non_callable_raises(self):
        """Missing args or non-callable run_lambda raises TypeError."""
        with self.assertRaises(TypeError):
            run_and_read_all(lambda c: (0, "", ""))
        with self.assertRaises(TypeError):
            run_and_read_all("not_callable", "cmd")


if __name__ == "__main__":
    run_tests()
