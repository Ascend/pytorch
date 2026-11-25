# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.run_and_parse_first_match 接口功能正确性
API 名称：torch.utils.collect_env.run_and_parse_first_match
API 签名：torch.utils.collect_env.run_and_parse_first_match(run_lambda, command, regex)

覆盖维度表：
| 覆盖维度         | 说明                                       | 覆盖情况                                               |
|------------------|--------------------------------------------|--------------------------------------------------|
| 空/非空          | 空 stdout 与非空 stdout 匹配行为           | 已覆盖（no_match_returns_none、matches_group1）        |
| 枚举选项         | 接口枚举类参数覆盖                         | 未覆盖（接口无枚举类参数）                             |
| 参数类型         | 字符串正则与编译后 re.Pattern 对象         | 已覆盖（matches_group1、compiled_regex）               |
| 传参与不传参     | 可选默认参数省略与显式传入                 | 未覆盖（run_lambda/command/regex 均为必填）            |
| 等价类/边界值    | 多行首匹配、无匹配、非零退出码             | 已覆盖（first_match_only、nonzero_rc_returns_none）    |
| 正常传参场景     | 正常执行命令并提取捕获组 1                 | 已覆盖（matches_group1、first_match_only、compiled_regex） |
| 异常传参场景     | 无捕获组正则与缺失必填参数                 | 已覆盖（no_capture_group_raises、missing_arg_raises）  |
| 混合设备类型     | 多 Tensor 输入时，NPU/CPU 混合设备输入场景 | 未覆盖（纯主机命令输出解析函数，不接收 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：接口无枚举类参数。
- 传参与不传参：run_lambda、command、regex 均为必填位置参数，无可选默认参数。
- 混合设备类型：纯主机命令输出正则解析辅助函数，不涉及 Tensor 与设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import re
import unittest

import torch
from torch.utils.collect_env import run_and_parse_first_match

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
class TestRunAndParseFirstMatch(TestCase):
    """Test cases for torch.utils.collect_env.run_and_parse_first_match."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_matches_group1(self):
        """Extracts group(1) and forwards command to run_lambda."""
        seen = []

        def fake_run(cmd):
            seen.append(cmd)
            return (0, "gcc (Ubuntu 11.4.0) 11.4.0\n", "")

        res = run_and_parse_first_match(fake_run, "gcc --version", r"gcc (.*)")
        self.assertEqual(res, "(Ubuntu 11.4.0) 11.4.0")
        self.assertEqual(seen, ["gcc --version"])

    def test_first_match_only(self):
        """Returns only the first matching group(1) across multiple lines."""
        out = "version: 1.0\nversion: 2.0\n"
        res = run_and_parse_first_match(lambda c: (0, out, ""), "cmd", r"version: (\S+)")
        self.assertEqual(res, "1.0")

    def test_compiled_regex(self):
        """Supports compiled re.Pattern objects as regex argument."""
        pat = re.compile(r"release (\d+\.\d+)")
        res = run_and_parse_first_match(lambda c: (0, "cuda release 12.1 end", ""), "nvcc", pat)
        self.assertEqual(res, "12.1")

    def test_no_match_returns_none(self):
        """Returns None when regex does not match stdout or stdout is empty."""
        self.assertIsNone(
            run_and_parse_first_match(lambda c: (0, "unrelated", ""), "cmd", r"version: (.*)")
        )
        self.assertIsNone(
            run_and_parse_first_match(lambda c: (0, "", ""), "cmd", r"version: (.*)")
        )

    def test_nonzero_rc_returns_none(self):
        """Returns None when run_lambda returns a non-zero exit code."""
        res = run_and_parse_first_match(lambda c: (1, "version: 1.0", "err"), "cmd", r"version: (.*)")
        self.assertIsNone(res)

    def test_no_capture_group_raises(self):
        """A matching regex with no capture group raises IndexError on group(1)."""
        with self.assertRaises(IndexError):
            run_and_parse_first_match(lambda c: (0, "abc", ""), "cmd", r"abc")

    def test_missing_arg_raises(self):
        """Missing required arguments raise TypeError."""
        with self.assertRaises(TypeError):
            run_and_parse_first_match(lambda c: (0, "", ""), "cmd")


if __name__ == "__main__":
    run_tests()
