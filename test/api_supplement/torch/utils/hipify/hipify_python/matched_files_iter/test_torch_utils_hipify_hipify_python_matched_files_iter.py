# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.matched_files_iter 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.matched_files_iter
API 签名：matched_files_iter(root_path, includes=(), ignores=(), extensions=(),
         out_of_place_only=False, is_pytorch_extension=False) -> Iterator[str]

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空目录、默认空 includes 与非空目录   | 已覆盖（empty_directory、default_args_empty、      |
|                  |                                      | extension_and_ignore_filter）                      |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 路径字符串与元组过滤参数             | 已覆盖（extension_and_ignore_filter、              |
|                  |                                      | multiple_extensions）                              |
| 传参与不传参     | 可选参数默认省略与显式传参           | 已覆盖（default_args_empty、                       |
|                  |                                      | extension_and_ignore_filter）                      |
| 等价类/边界值    | 多扩展名匹配、忽略规则与空目录       | 已覆盖（multiple_extensions、empty_directory、     |
|                  |                                      | extension_and_ignore_filter）                      |
| 正常传参场景     | 按 includes/ignores/extensions 过滤  | 已覆盖（extension_and_ignore_filter、              |
|                  |                                      | multiple_extensions）                              |
| 异常传参场景     | 缺失 root_path 抛出 TypeError        | 已覆盖（missing_arg_raises）                       |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯文件系统遍历工具，无 Tensor 输入）       |

未覆盖项及原因：
- 枚举选项：该接口参数为目录路径、匹配模式与布尔标志，无枚举类参数。
- 混合设备类型：该接口为纯文件系统遍历迭代器，不涉及 Tensor 或设备迁移。

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
from torch.utils.hipify.hipify_python import matched_files_iter

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
class TestHipifyMatchedFilesIter(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.matched_files_iter."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_extension_and_ignore_filter(self):
        """Filters files by includes, extensions, and ignores patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("kernel.cu", "ignore_me.cu", "helper.cpp", "notes.txt"):
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                    f.write("// test\n")

            matched = list(matched_files_iter(
                tmpdir,
                includes=("*",),
                ignores=("*ignore*",),
                extensions=(".cu",),
            ))
            basenames = [os.path.basename(p) for p in matched]
            self.assertEqual(basenames, ["kernel.cu"])
            self.assertNotIn("\\", matched[0])

    def test_multiple_extensions(self):
        """Matches multiple extensions when specified in extensions tuple."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("a.cu", "b.cpp", "c.txt"):
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                    f.write("")

            matched = list(matched_files_iter(
                tmpdir,
                includes=("*",),
                extensions=(".cu", ".cpp"),
            ))
            basenames = sorted(os.path.basename(p) for p in matched)
            self.assertEqual(basenames, ["a.cu", "b.cpp"])

    def test_default_args_empty(self):
        """Calling with only root_path yields no files because includes is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.cu"), "w", encoding="utf-8") as f:
                f.write("")
            matched = list(matched_files_iter(tmpdir))
            self.assertEqual(matched, [])

    def test_empty_directory(self):
        """Empty directory yields no files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            matched = list(matched_files_iter(tmpdir, includes=("*",), extensions=(".cu",)))
            self.assertEqual(matched, [])

    def test_missing_arg_raises(self):
        """Calling matched_files_iter without root_path raises TypeError."""
        with self.assertRaises(TypeError):
            list(matched_files_iter())


if __name__ == "__main__":
    run_tests()
