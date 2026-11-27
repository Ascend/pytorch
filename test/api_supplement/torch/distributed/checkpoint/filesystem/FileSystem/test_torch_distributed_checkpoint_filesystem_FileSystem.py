# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.filesystem.FileSystem 接口功能正确性
API 名称：torch.distributed.checkpoint.filesystem.FileSystem
API 签名：FileSystem()

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空目录 ls、空文件读写、非空二进制/文本流读写 | 已覆盖（test_path_mkdir_and_validate_checkpoint_id、test_stream_rename_and_rm_file） |
| 枚举选项 | 文件打开模式 rb/wb/r/w | 已覆盖（test_stream_rename_and_rm_file） |
| 参数类型 | str 与 pathlib.Path 路径输入 | 已覆盖（test_path_mkdir_and_validate_checkpoint_id、test_stream_rename_and_rm_file） |
| 传参与不传参 | 无参构造及各方法必选参数 | 已覆盖（全部用例） |
| 等价类/边界值 | 多级嵌套 mkdir、重复 mkdir 幂等、远程 URI 校验 | 已覆盖（test_path_mkdir_and_validate_checkpoint_id） |
| 正常传参场景 | 路径初始化/拼接、建目录、流读写、重命名、删文件 | 已覆盖（test_path_mkdir_and_validate_checkpoint_id、test_stream_rename_and_rm_file） |
| 异常传参场景 | 删除不存在文件、读取不存在文件、重命名不存在文件报错 | 已覆盖（test_error_paths） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（本地文件系统工具类，不接收 Tensor） |

未覆盖项及原因：
- 混合设备类型：FileSystem 为纯 Python 文件系统封装类，不涉及 Tensor 或设备迁移。

注意：本接口为本地文件系统封装类，所有临时目录均通过 tempfile.TemporaryDirectory 管理，
     测试结束后自动清理；按 API 语义使用文件内容、路径状态与异常断言验证。
"""
import os
import tempfile
import unittest
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


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestFileSystem(TestCase):
    def setUp(self):
        super().setUp()
        self.fs = FileSystem()

    def test_path_mkdir_and_validate_checkpoint_id(self):
        self.assertIsInstance(self.fs, FileSystemBase)
        with tempfile.TemporaryDirectory() as tmpdir:
            p_from_str = self.fs.init_path(tmpdir)
            p_from_path = self.fs.init_path(Path(tmpdir))
            self.assertIsInstance(p_from_str, Path)
            self.assertEqual(p_from_str, p_from_path)

            joined_str = self.fs.concat_path(tmpdir, "sub.distcp")
            self.assertEqual(joined_str, self.fs.concat_path(Path(tmpdir), "sub.distcp"))
            self.assertEqual(joined_str, Path(tmpdir) / "sub.distcp")

            nested = self.fs.concat_path(tmpdir, "a/b/c")
            self.assertFalse(self.fs.exists(nested))
            self.fs.mkdir(nested)
            self.assertTrue(self.fs.exists(nested))
            # Repeated mkdir on an existing directory must be idempotent.
            self.fs.mkdir(str(nested))
            self.assertEqual(self.fs.ls(nested), [])

            file_a = self.fs.concat_path(nested, "f1.bin")
            with self.fs.create_stream(file_a, "wb") as stream:
                stream.write(b"abc")
            self.assertEqual(self.fs.ls(str(nested)), [str(file_a)])

            sub_ckpt = Path(tmpdir) / "ckpt_0"
            self.assertTrue(FileSystem.validate_checkpoint_id(sub_ckpt))
            self.assertTrue(FileSystem.validate_checkpoint_id(str(sub_ckpt)))
            self.assertTrue(self.fs.validate_checkpoint_id(tmpdir))
        for uri in ("s3://bucket/ckpt", "hdfs://namenode/ckpt"):
            self.assertFalse(FileSystem.validate_checkpoint_id(uri))

    def test_stream_rename_and_rm_file(self):
        cases = (
            ("payload.bin", "wb", "rb", b"\x00\x01\x02checkpoint_bytes", True),
            ("empty.bin", "wb", "rb", b"", False),
            ("note.txt", "w", "r", "hello dcp", False),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for name, w_mode, r_mode, payload, write_as_str in cases:
                path = self.fs.concat_path(tmpdir, name)
                with self.fs.create_stream(str(path) if write_as_str else path, w_mode) as out_s:
                    out_s.write(payload)
                with self.fs.create_stream(path, r_mode) as in_s:
                    self.assertEqual(in_s.read(), payload)

            src = self.fs.concat_path(tmpdir, "payload.bin")
            dst = self.fs.concat_path(tmpdir, "dst.bin")
            self.fs.rename(str(src), dst)
            self.assertFalse(self.fs.exists(src))
            self.assertTrue(self.fs.exists(dst))
            with self.fs.create_stream(dst, "rb") as stream:
                self.assertEqual(stream.read(), b"\x00\x01\x02checkpoint_bytes")

            self.fs.rm_file(str(dst))
            self.assertFalse(self.fs.exists(dst))

    def test_error_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = self.fs.concat_path(tmpdir, "does_not_exist.bin")
            with self.assertRaises(FileNotFoundError):
                self.fs.rm_file(missing)
            with self.assertRaises(FileNotFoundError):
                self.fs.rename(missing, self.fs.concat_path(tmpdir, "target.bin"))
            with self.assertRaises(FileNotFoundError):
                with self.fs.create_stream(missing, "rb"):
                    pass


if __name__ == "__main__":
    run_tests()
