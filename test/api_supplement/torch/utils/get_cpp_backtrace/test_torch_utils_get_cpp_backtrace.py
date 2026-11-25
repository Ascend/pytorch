# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.get_cpp_backtrace 接口功能正确性
API 名称：torch.utils.get_cpp_backtrace
API 签名：torch.utils.get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=64) -> str

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                           |
|---------------|--------------------------------------|----------------------------------------------------|
| 空/非空       | 最大帧数为 0 与默认非空回溯          | 已覆盖（zero_max_frames、default_call）            |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（参数均为整数帧数，无枚举选项）             |
| 参数类型      | 合法 int 与非法 str 类型参数         | 已覆盖（explicit_args、invalid_arg_type_raises）   |
| 传参与不传参  | 默认无参、位置参数与关键字参数调用   | 已覆盖（default_call、explicit_args、keyword_args） |
| 等价类/边界值 | maximum_number_of_frames=0 及小/大值 | 已覆盖（zero_max_frames、explicit_args）           |
| 正常传参场景  | 获取当前线程 C++ 调用栈字符串        | 已覆盖（default_call、explicit_args、keyword_args） |
| 异常传参场景  | 非法类型与多余位置参数抛出 TypeError | 已覆盖（invalid_arg_type_raises、extra_arg_raises） |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯主机侧 C++ 栈回溯接口，无 Tensor 输入）  |

未覆盖项及原因：
- 枚举选项：frames_to_skip 与 maximum_number_of_frames 均为整数参数，无枚举选项。
- 混合设备类型：接口仅返回当前线程 C++ 栈回溯字符串，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch

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
class TestGetCppBacktrace(TestCase):
    """Test cases for torch.utils.get_cpp_backtrace."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_default_call(self):
        """Default call returns a string representation of the C++ backtrace."""
        bt = torch.utils.get_cpp_backtrace()
        self.assertIsInstance(bt, str)

    def test_explicit_args(self):
        """Explicit positional frames_to_skip and maximum_number_of_frames work."""
        bt_small = torch.utils.get_cpp_backtrace(1, 10)
        bt_large = torch.utils.get_cpp_backtrace(0, 64)
        self.assertIsInstance(bt_small, str)
        self.assertIsInstance(bt_large, str)
        # Smaller frame limit with skip should not exceed larger unfiltered trace.
        self.assertLessEqual(len(bt_small.splitlines()), max(len(bt_large.splitlines()), 12))

    def test_keyword_args(self):
        """Keyword arguments frames_to_skip and maximum_number_of_frames are accepted."""
        bt = torch.utils.get_cpp_backtrace(frames_to_skip=1, maximum_number_of_frames=5)
        self.assertIsInstance(bt, str)

    def test_zero_max_frames(self):
        """Setting maximum_number_of_frames=0 returns a string with no extra frames."""
        bt_zero = torch.utils.get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=0)
        bt_default = torch.utils.get_cpp_backtrace()
        self.assertIsInstance(bt_zero, str)
        self.assertLessEqual(len(bt_zero), len(bt_default))

    def test_invalid_arg_type_raises(self):
        """Passing non-integer arguments raises TypeError."""
        with self.assertRaises(TypeError):
            torch.utils.get_cpp_backtrace(frames_to_skip="bad")
        with self.assertRaises(TypeError):
            torch.utils.get_cpp_backtrace(maximum_number_of_frames="bad")

    def test_extra_arg_raises(self):
        """Passing too many positional arguments raises TypeError."""
        with self.assertRaises(TypeError):
            torch.utils.get_cpp_backtrace(0, 10, 99)


if __name__ == "__main__":
    run_tests()
