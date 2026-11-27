# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id 接口功能正确性
API 名称：torch.distributed.checkpoint.StorageWriter.validate_checkpoint_id
API 签名：StorageWriter.validate_checkpoint_id(cls, checkpoint_id) -> bool

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 合法路径字符串、Path 与远程 URI | 已覆盖（test_filesystem_writer_valid_and_invalid_ids、test_custom_subclass） |
| 枚举选项 | 不涉及枚举参数 | 未覆盖（接口仅接收 checkpoint_id 路径或标识参数） |
| 参数类型 | str 与 pathlib.Path (os.PathLike) | 已覆盖（test_filesystem_writer_valid_and_invalid_ids、test_custom_subclass） |
| 传参与不传参 | 必选参数显式传入与缺参报错 | 已覆盖（test_abstract_contract_and_missing_arg） |
| 等价类/边界值 | 本地可写目录、URI 协议前缀、自定义前缀 | 已覆盖（test_filesystem_writer_valid_and_invalid_ids、test_custom_subclass） |
| 正常传参场景 | 抽象方法调用、自定义与内置子类校验 | 已覆盖（全部 3 项用例） |
| 异常传参场景 | 缺失抽象方法实例化报错、缺少必选参数报错 | 已覆盖（test_abstract_contract_and_missing_arg） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯 Python 存储抽象接口，不接收 Tensor） |

未覆盖项及原因：
- 枚举选项：接口仅接收 checkpoint_id 参数，无枚举型入参。
- 混合设备类型：接口为纯 Python 存储抽象方法，不涉及 Tensor 或跨设备操作。

注意：本接口为纯 Python 分布式存储抽象类方法，无浮点 Tensor 输出，
     按 API 语义使用抽象类契约、返回值布尔结果与异常断言验证。
"""
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemWriter, StorageWriter

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


class _IncompleteWriter(StorageWriter):
    def reset(self, checkpoint_id=None):
        return None

    def set_up_storage_writer(self, is_coordinator, *args, **kwargs):
        return None

    def prepare_local_plan(self, plan):
        return plan

    def prepare_global_plan(self, plans):
        return plans

    def write_data(self, plan, planner):
        return None

    def finish(self, metadata, results):
        return None


class _PrefixStorageWriter(_IncompleteWriter):
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id):
        return str(checkpoint_id).startswith("mem://")


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestStorageWriterValidateCheckpointId(TestCase):
    def test_abstract_contract_and_missing_arg(self):
        self.assertIn("validate_checkpoint_id", StorageWriter.__abstractmethods__)
        self.assertIsNone(StorageWriter.validate_checkpoint_id("/tmp/ckpt"))
        with self.assertRaises(TypeError):
            _IncompleteWriter()
        with self.assertRaises(TypeError):
            StorageWriter.validate_checkpoint_id()

    def test_custom_subclass(self):
        writer = _PrefixStorageWriter()
        self.assertIsInstance(writer, StorageWriter)
        self.assertTrue(_PrefixStorageWriter.validate_checkpoint_id("mem://store/ckpt"))
        self.assertTrue(writer.validate_checkpoint_id("mem://step_2"))
        for bad_id in ("/local/path/ckpt", Path("/local/path/ckpt")):
            self.assertFalse(_PrefixStorageWriter.validate_checkpoint_id(bad_id))

    def test_filesystem_writer_valid_and_invalid_ids(self):
        self.assertTrue(issubclass(FileSystemWriter, StorageWriter))
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt_step_1"
            for valid_id in (ckpt_path, str(ckpt_path), tmpdir):
                self.assertTrue(FileSystemWriter.validate_checkpoint_id(valid_id))
        for uri in ("s3://bucket/ckpt", "gcs://bucket/ckpt", "https://example.com/ckpt"):
            self.assertFalse(FileSystemWriter.validate_checkpoint_id(uri))


if __name__ == "__main__":
    run_tests()
