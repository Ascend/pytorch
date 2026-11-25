# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.check_release_file 接口功能正确性
API 名称：torch.utils.collect_env.check_release_file
API 签名：torch.utils.collect_env.check_release_file(run_lambda)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空输出与非空 release 文件内容        | 已覆盖（empty_output_returns_none、matches_pretty_name） |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 可调用函数与非可调用字符串参数       | 已覆盖（matches_pretty_name、non_callable_raises） |
| 传参与不传参     | 必选回调传参与缺参                   | 已覆盖（matches_pretty_name、missing_arg_raises）  |
| 等价类/边界值    | 多行首匹配、无匹配、非零返回码       | 已覆盖（multiline_returns_first_match、no_match_returns_none 等） |
| 正常传参场景     | 解析 PRETTY_NAME 与命令调用校验      | 已覆盖（matches_pretty_name、multiline_returns_first_match 等） |
| 异常传参场景     | 缺参与非可调用对象触发 TypeError     | 已覆盖（missing_arg_raises、non_callable_raises）  |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 枚举选项：接口仅接收单个 run_lambda 回调，无枚举选项。
- 混合设备类型：环境信息采集辅助函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.collect_env import check_release_file

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
class TestCheckReleaseFile(TestCase):
    """Test cases for torch.utils.collect_env.check_release_file."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_matches_pretty_name(self):
        """Extracts PRETTY_NAME value and invokes run_lambda with release cat command."""
        seen_cmds = []

        def fake_run(cmd):
            seen_cmds.append(cmd)
            return (0, 'NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 22.04.3 LTS"\n', "")

        res = check_release_file(fake_run)
        self.assertEqual(res, "Ubuntu 22.04.3 LTS")
        self.assertEqual(seen_cmds, ["cat /etc/*-release"])

    def test_multiline_returns_first_match(self):
        """Returns the first PRETTY_NAME match when multiple release files are concatenated."""
        out = 'PRETTY_NAME="First OS"\nPRETTY_NAME="Second OS"\n'
        res = check_release_file(lambda cmd: (0, out, ""))
        self.assertEqual(res, "First OS")

    def test_no_match_returns_none(self):
        """Returns None when output does not contain PRETTY_NAME."""
        res = check_release_file(lambda cmd: (0, 'NAME="Ubuntu"\nVERSION="22.04"\n', ""))
        self.assertIsNone(res)

    def test_empty_output_returns_none(self):
        """Returns None when command succeeds with empty stdout."""
        res = check_release_file(lambda cmd: (0, "", ""))
        self.assertIsNone(res)

    def test_nonzero_rc_returns_none(self):
        """Returns None when run_lambda returns a non-zero exit code."""
        res = check_release_file(lambda cmd: (1, 'PRETTY_NAME="Ignored"\n', "not found"))
        self.assertIsNone(res)

    def test_missing_arg_raises(self):
        """Calling check_release_file without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            check_release_file()

    def test_non_callable_raises(self):
        """Passing a non-callable run_lambda raises TypeError."""
        with self.assertRaises(TypeError):
            check_release_file("not_callable")


if __name__ == "__main__":
    run_tests()
