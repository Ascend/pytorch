# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.file_add_header 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.file_add_header
API 签名：torch.utils.hipify.hipify_python.file_add_header(filepath, header) -> None

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                             |
|---------------|--------------------------------------|------------------------------------------------------|
| 空/非空       | 空文件与已有内容的非空源文件         | 已覆盖（empty_file、prepend_quoted_header）          |
| 枚举选项      | 枚举型参数选项                       | 未覆盖（参数为文件路径与头文件名，无枚举选项）       |
| 参数类型      | str 文件路径与带/不带尖括号头文件    | 已覆盖（prepend_quoted_header、prepend_angle_header） |
| 传参与不传参  | 完整传参与缺少必需参数               | 已覆盖（prepend_quoted_header、missing_args_raises） |
| 等价类/边界值 | 引号头文件、尖括号头文件与空文件边界 | 已覆盖（prepend_quoted_header、prepend_angle_header、empty_file） |
| 正常传参场景  | 文件首部插入 #include 指令并保留正文 | 已覆盖（prepend_quoted_header、prepend_angle_header、empty_file） |
| 异常传参场景  | 文件不存在与缺参异常                 | 已覆盖（nonexistent_file_raises、missing_args_raises） |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯文件文本读写工具，无 Tensor 输入）         |

未覆盖项及原因：
- 枚举选项：参数 filepath 与 header 均为字符串，无枚举类参数。
- 混合设备类型：接口仅对本地文本文件执行头部插入操作，无 Tensor 输入，不涉及设备迁移。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import os
import tempfile
import unittest

import torch
from torch.utils.hipify.hipify_python import file_add_header

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
class TestHipifyFileAddHeader(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.file_add_header."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_prepend_quoted_header(self):
        """Plain header name is wrapped in quotes and prepended to existing content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sample.cpp")
            original = "int main() { return 0; }\n"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(original)

            ret = file_add_header(fpath, "hip/hip_runtime.h")
            self.assertIsNone(ret)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, '#include "hip/hip_runtime.h" \n' + original)

    def test_prepend_angle_header(self):
        """Angle-bracketed header is preserved without extra quotes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sample.cpp")
            original = "void foo();\n"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(original)

            file_add_header(fpath, "<vector>")
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "#include <vector> \n" + original)

    def test_empty_file(self):
        """Adding a header to an empty file writes only the include directive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "empty.h")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("")

            file_add_header(fpath, "my_defs.h")
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, '#include "my_defs.h" \n')

    def test_nonexistent_file_raises(self):
        """Nonexistent file path raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "does_not_exist.cpp")
            with self.assertRaises(FileNotFoundError):
                file_add_header(missing, "header.h")

    def test_missing_args_raises(self):
        """Calling file_add_header without required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            file_add_header()
        with self.assertRaises(TypeError):
            file_add_header("only_path.cpp")


if __name__ == "__main__":
    run_tests()
