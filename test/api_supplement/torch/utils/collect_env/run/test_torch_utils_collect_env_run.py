# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.run 接口功能正确性
API 名称：torch.utils.collect_env.run
API 签名：torch.utils.collect_env.run(command)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 静默命令空输出与非空标准输出         | 已覆盖（empty_output、list_command_success）       |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | list 命令、str 命令与非法 int 参数   | 已覆盖（list_command_success、str_command_success、invalid_arg_raises） |
| 传参与不传参     | 必选 command 传参与缺参              | 已覆盖（list_command_success、missing_arg_raises） |
| 等价类/边界值    | 首尾空白裁剪、空输出及非零退出码     | 已覆盖（strips_whitespace、empty_output、failing_command） |
| 正常传参场景     | 列表/字符串命令执行与三元组返回值    | 已覆盖（list_command_success、str_command_success、failing_command 等） |
| 异常传参场景     | 缺参与非法整型参数触发 TypeError     | 已覆盖（missing_arg_raises、invalid_arg_raises）   |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 枚举选项：接口仅接收命令字符串或列表，无枚举选项。
- 混合设备类型：子进程命令执行辅助函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import sys
import unittest

import torch
from torch.utils.collect_env import run

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
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCollectEnvRun(TestCase):
    """Test cases for torch.utils.collect_env.run."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_list_command_success(self):
        """Running a list command returns (0, 'ok', '') with expected tuple types."""
        res = run([sys.executable, "-c", "print('ok')"])
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 3)
        rc, out, err = res
        self.assertIsInstance(rc, int)
        self.assertIsInstance(out, str)
        self.assertIsInstance(err, str)
        self.assertEqual((rc, out, err), (0, "ok", ""))

    def test_str_command_success(self):
        """Running a string shell command returns expected stdout."""
        cmd = f'"{sys.executable}" -c "print(42)"'
        rc, out, err = run(cmd)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "42")
        self.assertEqual(err, "")

    def test_strips_whitespace(self):
        """Leading and trailing whitespace is stripped from stdout and stderr."""
        rc, out, err = run([sys.executable, "-c", "print('  hello  ')"])
        self.assertEqual(rc, 0)
        self.assertEqual(out, "hello")
        self.assertEqual(err, "")

    def test_empty_output(self):
        """Silent command returns (0, '', '')."""
        self.assertEqual(run([sys.executable, "-c", "pass"]), (0, "", ""))

    def test_failing_command(self):
        """Failing command returns non-zero exit code and captured stderr."""
        code = "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"
        rc, out, err = run([sys.executable, "-c", code])
        self.assertEqual(rc, 3)
        self.assertEqual(out, "")
        self.assertEqual(err, "boom")

    def test_missing_arg_raises(self):
        """Calling run() without arguments raises TypeError."""
        with self.assertRaises(TypeError):
            run()

    def test_invalid_arg_raises(self):
        """Passing a non-command integer raises TypeError."""
        with self.assertRaises(TypeError):
            run(12345)


if __name__ == "__main__":
    run_tests()
