# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.collect_env.main 接口功能正确性
API 名称：torch.utils.collect_env.main
API 签名：torch.utils.collect_env.main()

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空与非空环境信息输出                 | 已覆盖（empty_env_info_output、prints_header_and_env_info） |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（见下）                                     |
| 参数类型         | 输入参数类型                         | 未覆盖（见下）                                     |
| 传参与不传参     | 无参调用与多余位置/关键字参数        | 已覆盖（prints_header_and_env_info、unexpected_arg_raises） |
| 等价类/边界值    | 环境信息为空字符串边界               | 已覆盖（empty_env_info_output）                    |
| 正常传参场景     | 输出标题与环境信息并返回 None        | 已覆盖（prints_header_and_env_info、empty_env_info_output） |
| 异常传参场景     | 传入多余位置或关键字参数报 TypeError | 已覆盖（unexpected_arg_raises）                    |
| 混合设备类型     | 多 Tensor 输入时 NPU/CPU 混合场景    | 未覆盖（见下）                                     |

未覆盖项及原因：
- 枚举选项：无参入口函数，无枚举类型参数。
- 参数类型：接口不接收外部参数，多余参数已在异常传参场景验证。
- 混合设备类型：命令行环境采集入口函数，不接收 Tensor 输入。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

import torch
import torch.utils.collect_env as collect_env
from torch.utils.collect_env import main

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
class TestCollectEnvMain(TestCase):
    """Test cases for torch.utils.collect_env.main."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_prints_header_and_env_info(self):
        """main() prints header and get_pretty_env_info() output and returns None."""
        buf = io.StringIO()
        sample = "PyTorch version: 2.15.0\nOS: Linux"
        with patch.object(collect_env, "get_pretty_env_info", return_value=sample) as mock_fn:
            with patch.object(collect_env.os.path, "exists", return_value=False):
                with contextlib.redirect_stdout(buf):
                    ret = main()
        self.assertIsNone(ret)
        mock_fn.assert_called_once_with()
        output = buf.getvalue()
        self.assertIn("Collecting environment information...", output)
        self.assertIn(sample, output)

    def test_empty_env_info_output(self):
        """main() handles empty string from get_pretty_env_info without error."""
        buf = io.StringIO()
        with patch.object(collect_env, "get_pretty_env_info", return_value=""):
            with patch.object(collect_env.os.path, "exists", return_value=False):
                with contextlib.redirect_stdout(buf):
                    ret = main()
        self.assertIsNone(ret)
        self.assertIn("Collecting environment information...", buf.getvalue())

    def test_unexpected_arg_raises(self):
        """Passing unexpected positional or keyword arguments raises TypeError."""
        with self.assertRaises(TypeError):
            main("extra")
        with self.assertRaises(TypeError):
            main(verbose=True)


if __name__ == "__main__":
    run_tests()
