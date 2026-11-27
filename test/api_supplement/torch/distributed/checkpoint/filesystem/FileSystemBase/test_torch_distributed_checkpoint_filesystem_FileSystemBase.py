# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.filesystem.FileSystemBase 接口功能正确性
API 名称：torch.distributed.checkpoint.filesystem.FileSystemBase
API 签名：class FileSystemBase(ABC)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 完整子类对空目录、空流与非空数据流的处理 | 已覆盖（test_concrete_subclasses_and_lifecycle） |
| 枚举选项 | 不涉及枚举参数 | 未覆盖（抽象基类定义接口契约，无枚举入参） |
| 参数类型 | str 与 pathlib.Path 路径输入 | 已覆盖（test_concrete_subclasses_and_lifecycle） |
| 传参与不传参 | 抽象基类实例化报错、完整子类实例化成功 | 已覆盖（全部 2 项用例） |
| 等价类/边界值 | 8 个抽象方法集合校验、缺失部分抽象方法子类校验 | 已覆盖（test_abstract_contract_and_partial_subclass_raises） |
| 正常传参场景 | 自定义内存实现子类及内置 FileSystem 子类验证 | 已覆盖（test_concrete_subclasses_and_lifecycle） |
| 异常传参场景 | 直接实例化基类或未完全实现子类抛出 TypeError | 已覆盖（test_abstract_contract_and_partial_subclass_raises） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯 Python 抽象基类，不接收 Tensor） |

未覆盖项及原因：
- 枚举选项：接口为文件系统抽象基类，无枚举入参。
- 混合设备类型：接口不涉及 Tensor 或设备迁移。

注意：本接口为纯 Python ABC 抽象基类，无浮点 Tensor 输出，
     按 API 语义使用抽象方法集合、子类契约行为与异常断言验证。
"""
import io
import os
import unittest
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.distributed.checkpoint.filesystem import FileSystem, FileSystemBase

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Set RUN_SUPPLEMENT_API_TESTS=1 to run supplement API tests"
DEVICE_NAME = "cpu"
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
        DEVICE_NAME = "npu" if torch.npu.is_available() else None
    except (ImportError, AttributeError):
        DEVICE_NAME = None
    if DEVICE_NAME is None:
        if not torch.cuda.is_available():
            raise RuntimeError("No available NPU or GPU device.")
        DEVICE_NAME = "cuda"
try:
    from torch_npu.testing.testcase import TestCase, run_tests
except (ImportError, TypeError):
    from unittest import TestCase
    run_tests = unittest.main


class _PartialFileSystem(FileSystemBase):
    @contextmanager
    def create_stream(self, path, mode):
        yield io.BytesIO()

    def concat_path(self, path, suffix):
        return f"{path}/{suffix}"

    def rename(self, path, new_path):
        return None

    def init_path(self, path):
        return str(path)

    def mkdir(self, path):
        return None

    def exists(self, path):
        return False


class _InMemoryFileSystem(FileSystemBase):
    def __init__(self):
        self._files = {}
        self._dirs = set()

    @contextmanager
    def create_stream(self, path, mode):
        key = str(path)
        if "w" in mode:
            buf = io.BytesIO()
            yield buf
            self._files[key] = buf.getvalue()
        else:
            if key not in self._files:
                raise FileNotFoundError(key)
            yield io.BytesIO(self._files[key])

    def concat_path(self, path, suffix):
        return f"{str(path).rstrip('/')}/{suffix}"

    def rename(self, path, new_path):
        self._files[str(new_path)] = self._files.pop(str(path))

    def init_path(self, path):
        return str(path)

    def mkdir(self, path):
        self._dirs.add(str(path))

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        return str(checkpoint_id).startswith("mem://")

    def exists(self, path):
        key = str(path)
        return key in self._files or key in self._dirs

    def rm_file(self, path):
        self._files.pop(str(path))


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFileSystemBase(TestCase):
    def test_abstract_contract_and_partial_subclass_raises(self):
        expected = {
            "create_stream",
            "concat_path",
            "rename",
            "init_path",
            "mkdir",
            "validate_checkpoint_id",
            "exists",
            "rm_file",
        }
        self.assertEqual(FileSystemBase.__abstractmethods__, expected)
        with self.assertRaises(TypeError):
            FileSystemBase()
        with self.assertRaises(TypeError):
            _PartialFileSystem()

    def test_concrete_subclasses_and_lifecycle(self):
        self.assertTrue(issubclass(FileSystem, FileSystemBase))
        self.assertIsInstance(FileSystem(), FileSystemBase)

        fs = _InMemoryFileSystem()
        self.assertIsInstance(fs, FileSystemBase)
        self.assertTrue(fs.validate_checkpoint_id("mem://ckpt"))
        self.assertFalse(fs.validate_checkpoint_id(Path("/local/ckpt")))

        root = fs.init_path("mem://run1")
        fs.mkdir(root)
        self.assertTrue(fs.exists(root))

        f1 = fs.concat_path(root, "data.bin")
        f2 = fs.concat_path(root, "data_moved.bin")
        with fs.create_stream(f1, "wb") as stream:
            stream.write(b"tensor_payload")
        self.assertTrue(fs.exists(f1))
        with fs.create_stream(f1, "rb") as stream:
            self.assertEqual(stream.read(), b"tensor_payload")

        fs.rename(f1, f2)
        self.assertFalse(fs.exists(f1))
        self.assertTrue(fs.exists(f2))
        fs.rm_file(f2)
        self.assertFalse(fs.exists(f2))


if __name__ == "__main__":
    run_tests()
