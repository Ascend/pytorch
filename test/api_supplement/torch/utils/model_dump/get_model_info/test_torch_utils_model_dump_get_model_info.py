# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.utils.model_dump.get_model_info 接口功能正确性
API 名称：torch.utils.model_dump.get_model_info
API 签名：torch.utils.model_dump.get_model_info(path_or_file, title=None,
         extra_file_size_limit=DEFAULT_EXTRA_FILE_SIZE_LIMIT)

覆盖维度表：
| 覆盖维度         | 说明                                 | 覆盖情况                                           |
|------------------|--------------------------------------|----------------------------------------------------|
| 空/非空          | 有效模型文件与不存在路径             | 已覆盖（basic_path_str、nonexistent_path_raises）  |
| 枚举选项         | 枚举类参数选项                       | 未覆盖（无枚举类参数）                             |
| 参数类型         | 文件路径字符串与 io.BytesIO 内存流   | 已覆盖（basic_path_str、bytesio_input）            |
| 传参与不传参     | 可选参数省略/显式传入与缺参          | 已覆盖（basic_path_str、                           |
|                  |                                      | custom_title_and_size_limit、missing_arg_raises）  |
| 等价类/边界值    | 自定义 title/size_limit 与内存流默认 | 已覆盖（custom_title_and_size_limit、              |
|                  | 标题 buffer                          | bytesio_input）                                    |
| 正常传参场景     | 解析 TorchScript 文件与流元信息字典  | 已覆盖（basic_path_str、                           |
|                  |                                      | custom_title_and_size_limit、bytesio_input）       |
| 异常传参场景     | 不存在路径与缺失必填参数异常         | 已覆盖（nonexistent_path_raises、                  |
|                  |                                      | missing_arg_raises）                               |
| 混合设备类型     | 多 Tensor 输入 NPU/CPU 混合设备场景  | 未覆盖（纯序列化模型包解析工具，无混合设备输入）   |

未覆盖项及原因：
- 枚举选项：该接口参数为路径/文件流、标题字符串与字节上限整数，无枚举类参数。
- 混合设备类型：该接口解析已保存的 TorchScript zip 包元信息，不涉及多 Tensor 混合设备运算。

注意：对可在 CPU/NPU 两侧执行的**非图模式**浮点 Tensor 输出，以 CPU 结果为 Golden；
     将待比较的输出移至 CPU 后使用 torch.testing.assert_close 进行浮点对照；
     其他结果按 API 语义使用结构、类型、副作用或异常断言。
     图模式类（普通 Tensor / 配置 / 注册表 / IR 或图结构 / 编译入口）的 Golden 规则见
     references/GRAPH_API_UT.md，优先于本通用规则。
"""
import io
import os
import tempfile
import unittest

import torch
from torch.utils.model_dump import get_model_info

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
class TestGetModelInfo(TestCase):
    """Test cases for torch.utils.model_dump.get_model_info."""

    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self._tmpdir = tempfile.TemporaryDirectory()
        self.scripted = torch.jit.script(torch.nn.Linear(4, 2))
        self.model_path = os.path.join(self._tmpdir.name, "tiny_linear.pt")
        torch.jit.save(self.scripted, self.model_path)

    def tearDown(self):
        self._tmpdir.cleanup()
        super().tearDown()

    def test_basic_path_str(self):
        """get_model_info parses a saved TorchScript file path and populates metadata."""
        info = get_model_info(self.model_path)
        self.assertIsInstance(info, dict)
        self.assertIn("model", info)
        model = info["model"]
        self.assertEqual(model["title"], self.model_path)
        self.assertGreater(model["file_size"], 0)
        self.assertIsInstance(model["zip_files"], list)
        self.assertGreater(len(model["zip_files"]), 0)
        self.assertIn("model_data", model)
        self.assertIn("version", model)

    def test_custom_title_and_size_limit(self):
        """Explicit title overrides default path title and respects extra_file_size_limit."""
        info = get_model_info(self.model_path, title="MyModel", extra_file_size_limit=1024)
        self.assertEqual(info["model"]["title"], "MyModel")
        self.assertGreater(info["model"]["file_size"], 0)

    def test_bytesio_input(self):
        """get_model_info accepts an in-memory io.BytesIO stream with default title 'buffer'."""
        buf = io.BytesIO()
        torch.jit.save(self.scripted, buf)
        buf.seek(0)
        info = get_model_info(buf)
        self.assertEqual(info["model"]["title"], "buffer")
        self.assertGreater(info["model"]["file_size"], 0)
        self.assertGreater(len(info["model"]["zip_files"]), 0)

    def test_nonexistent_path_raises(self):
        """Nonexistent file path raises FileNotFoundError / OSError."""
        bad_path = os.path.join(self._tmpdir.name, "does_not_exist.pt")
        with self.assertRaises((FileNotFoundError, OSError)):
            get_model_info(bad_path)

    def test_missing_arg_raises(self):
        """Calling get_model_info without path_or_file raises TypeError."""
        with self.assertRaises(TypeError):
            get_model_info()


if __name__ == "__main__":
    run_tests()
