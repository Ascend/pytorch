# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.match_extensions 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.match_extensions
API 签名：torch.utils.hipify.hipify_python.match_extensions(filename: str, extensions: Iterable) -> bool

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空文件名、空扩展名集合与非空输入     | 已覆盖（empty_extensions、empty_filename、         |
|                  |                                      | matching_extensions）                              |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | tuple/list/generator 等可迭代扩展名  | 已覆盖（iterable_types、matching_extensions）      |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（matching_extensions、missing_args_raises） |
| 等价类/边界值    | 大小写敏感与非精确后缀边界           | 已覆盖（case_sensitive_extension、                 |
|                  |                                      | non_matching_extensions）                          |
| 正常传参场景     | 匹配与不匹配扩展名判定               | 已覆盖（matching_extensions、                      |
|                  |                                      | non_matching_extensions）                          |
| 异常传参场景     | 缺失必填参数抛出 TypeError           | 已覆盖（missing_args_raises）                      |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯字符串扩展名匹配工具，无 Tensor 输入）   |

未覆盖项及原因：
- 枚举选项：该接口参数为文件名字符串与扩展名可迭代对象，无枚举类参数。
- 混合设备类型：该接口为纯字符串后缀匹配函数，不涉及 Tensor 或设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import unittest

import torch
from torch.utils.hipify.hipify_python import match_extensions

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
class TestHipifyMatchExtensions(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.match_extensions."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_matching_extensions(self):
        """Filenames ending with a listed extension return True."""
        self.assertIs(match_extensions("kernel.cu", (".cu", ".cpp")), True)
        self.assertIs(match_extensions("dir/sub/module.cpp", (".cu", ".cpp")), True)

    def test_non_matching_extensions(self):
        """Filenames without a matching suffix return False."""
        self.assertIs(match_extensions("script.py", (".cu", ".cpp")), False)
        self.assertIs(match_extensions("kernel.cuh", (".cu", ".cpp")), False)

    def test_iterable_types(self):
        """Supports list and generator iterables for extensions."""
        self.assertIs(match_extensions("ops.cu", [".h", ".cu"]), True)
        self.assertIs(match_extensions("ops.h", (ext for ext in (".h", ".cu"))), True)

    def test_empty_extensions(self):
        """Empty extensions iterable always returns False."""
        self.assertIs(match_extensions("kernel.cu", ()), False)
        self.assertIs(match_extensions("kernel.cu", []), False)

    def test_empty_filename(self):
        """Empty filename with non-empty extensions returns False."""
        self.assertIs(match_extensions("", (".cu", ".cpp")), False)

    def test_case_sensitive_extension(self):
        """Extension matching is case-sensitive and requires exact suffix match."""
        self.assertIs(match_extensions("kernel.CU", (".cu",)), False)
        self.assertIs(match_extensions("kernel.cu.bak", (".cu",)), False)

    def test_missing_args_raises(self):
        """Calling match_extensions with missing required args raises TypeError."""
        with self.assertRaises(TypeError):
            match_extensions()
        with self.assertRaises(TypeError):
            match_extensions("kernel.cu")


if __name__ == "__main__":
    run_tests()
