# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.hipify.hipify_python.openf 接口功能正确性
API 名称：torch.utils.hipify.hipify_python.openf
API 签名：torch.utils.hipify.hipify_python.openf(filename, mode)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 空文件内容与非空文本读写             | 已覆盖（empty_file_roundtrip、write_and_read）     |
| 枚举选项         | 文件打开模式 r / w / a               | 已覆盖（write_and_read、append_mode）              |
| 参数类型         | 文件路径与模式字符串                 | 已覆盖（write_and_read、append_mode）              |
| 传参与不传参     | 必填参数传入与缺参                   | 已覆盖（write_and_read、missing_args_raises）      |
| 等价类/边界值    | 空文件往返与追加模式写入             | 已覆盖（empty_file_roundtrip、append_mode）        |
| 正常传参场景     | 正常写入、读取与追加文本             | 已覆盖（write_and_read、append_mode）              |
| 异常传参场景     | 不存在文件读模式与缺参异常           | 已覆盖（nonexistent_file_raises、                  |
|                  |                                      | missing_args_raises）                              |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯文件打开封装工具，无 Tensor 输入）       |

未覆盖项及原因：
- 混合设备类型：该接口为纯 Python 文件打开辅助函数，不涉及 Tensor 或设备迁移。

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
from torch.utils.hipify.hipify_python import openf

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
class TestHipifyOpenf(TestCase):
    """Test cases for torch.utils.hipify.hipify_python.openf."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME

    def test_write_and_read(self):
        """Writing text with mode='w' and reading back with mode='r' preserves content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "sample.txt")
            with openf(fpath, "w") as f:
                f.write("hello")
            with openf(fpath, "r") as f:
                content = f.read()
            self.assertEqual(content, "hello")

    def test_empty_file_roundtrip(self):
        """Empty content written and read back produces an empty string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "empty.txt")
            with openf(fpath, "w") as f:
                f.write("")
            with openf(fpath, "r") as f:
                content = f.read()
            self.assertEqual(content, "")

    def test_append_mode(self):
        """Appending with mode='a' adds text to existing file contents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "append.txt")
            with openf(fpath, "w") as f:
                f.write("hello")
            with openf(fpath, "a") as f:
                f.write(" world")
            with openf(fpath, "r") as f:
                content = f.read()
            self.assertEqual(content, "hello world")

    def test_nonexistent_file_raises(self):
        """Opening a nonexistent file in read mode raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "does_not_exist.txt")
            with self.assertRaises(FileNotFoundError):
                openf(missing, "r")

    def test_missing_args_raises(self):
        """Calling openf with missing required arguments raises TypeError."""
        with self.assertRaises(TypeError):
            openf()
        with self.assertRaises(TypeError):
            openf("dummy.txt")


if __name__ == "__main__":
    run_tests()
