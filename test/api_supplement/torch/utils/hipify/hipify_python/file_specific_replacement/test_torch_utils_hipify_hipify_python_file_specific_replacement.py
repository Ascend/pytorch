# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.file_specific_replacement 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.file_specific_replacement
API 签名：file_specific_replacement(filepath, search_string, replace_string, strict=False) -> None

覆盖维度表：
| 覆盖维度      | 说明                                 | 覆盖情况                                              |
|---------------|--------------------------------------|-------------------------------------------------------|
| 空/非空       | 无匹配保持不变与命中目标文本替换     | 已覆盖（no_match_unchanged、basic_replace）           |
| 枚举选项      | strict=False/True 布尔匹配模式       | 已覆盖（basic_replace、strict_word_boundary、no_match_unchanged） |
| 参数类型      | str 路径/替换串与 bool strict 参数   | 已覆盖（basic_replace、strict_word_boundary）         |
| 传参与不传参  | strict 默认值与显式传参              | 已覆盖（basic_replace、strict_word_boundary）         |
| 等价类/边界值 | 词边界匹配、无匹配与短串截断覆写     | 已覆盖（strict_word_boundary、no_match_unchanged、shorter_replacement_truncates） |
| 正常传参场景  | 原地替换文件内容并正确截断           | 已覆盖（basic_replace、strict_word_boundary、shorter_replacement_truncates） |
| 异常传参场景  | 文件不存在与缺参异常                 | 已覆盖（nonexistent_file_raises、missing_args_raises） |
| 混合设备类型  | 多 Tensor 输入时 NPU/CPU 混合        | 未覆盖（纯文件文本替换工具，无 Tensor 输入）          |

未覆盖项及原因：
- 混合设备类型：接口仅对磁盘文本文件执行字符串替换与截断，无 Tensor 输入，不涉及设备迁移。

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
from torch.utils.hipify.hipify_python import file_specific_replacement

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
class TestHipifyFileSpecificReplacement(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.file_specific_replacement."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_basic_replace(self):
        """Default non-strict mode replaces all substring occurrences in place."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "code.cu")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("cudaMalloc(p); cudaMallocHost(q);\n")

            ret = file_specific_replacement(fpath, "cudaMalloc", "hipMalloc")
            self.assertIsNone(ret)
            with open(fpath, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "hipMalloc(p); hipMallocHost(q);\n")

    def test_strict_word_boundary(self):
        """Strict mode replaces only whole-word matches respecting word boundaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "code.cu")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("cudaMalloc(p); cudaMallocHost(q);\n")

            file_specific_replacement(fpath, "cudaMalloc", "hipMalloc", strict=True)
            with open(fpath, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "hipMalloc(p); cudaMallocHost(q);\n")

    def test_no_match_unchanged(self):
        """File content remains unchanged when search_string is not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "code.cu")
            original = "int x = 1;\n"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(original)

            file_specific_replacement(fpath, "notPresent", "replaced", strict=False)
            file_specific_replacement(fpath, "notPresent", "replaced", strict=True)
            with open(fpath, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), original)

    def test_shorter_replacement_truncates(self):
        """Replacing with a shorter string truncates trailing bytes from the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "short.cu")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write("very_long_identifier_name")

            file_specific_replacement(fpath, "very_long_identifier_name", "x")
            with open(fpath, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "x")

    def test_nonexistent_file_raises(self):
        """Nonexistent file path raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "missing.cu")
            with self.assertRaises(FileNotFoundError):
                file_specific_replacement(missing, "a", "b")

    def test_missing_args_raises(self):
        """Calling file_specific_replacement with missing arguments raises TypeError."""
        with self.assertRaises(TypeError):
            file_specific_replacement()
        with self.assertRaises(TypeError):
            file_specific_replacement("a.cu", "foo")


if __name__ == "__main__":
    run_tests()
