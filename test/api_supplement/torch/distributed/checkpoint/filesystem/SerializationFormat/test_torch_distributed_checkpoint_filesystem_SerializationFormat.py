# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.filesystem.SerializationFormat 接口功能正确性
API 名称：torch.distributed.checkpoint.filesystem.SerializationFormat
API 签名：class SerializationFormat(Enum)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 不涉及容器或空 Tensor 输入 | 未覆盖（Enum 类型无空容器入参） |
| 枚举选项 | TORCH_SAVE 与 SAFETENSORS 全部枚举成员 | 已覆盖（test_members_lookup_and_identity） |
| 参数类型 | 按字符串值查找与按成员名下标查找 | 已覆盖（test_members_lookup_and_identity、test_invalid_lookup_raises） |
| 传参与不传参 | FileSystemWriter 默认与显式指定 serialization_format | 已覆盖（test_filesystem_writer_integration） |
| 等价类/边界值 | 成员相等性、不等性、哈希与集合/字典键兼容 | 已覆盖（test_members_lookup_and_identity） |
| 正常传参场景 | 枚举属性读取、查找及 Writer 集成 | 已覆盖（test_members_lookup_and_identity、test_filesystem_writer_integration） |
| 异常传参场景 | 非法值抛 ValueError、非法成员名抛 KeyError | 已覆盖（test_invalid_lookup_raises） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯 Python 枚举类，不接收 Tensor） |

未覆盖项及原因：
- 空/非空：SerializationFormat 为固定枚举类，无空/非空容器参数。
- 混合设备类型：纯 Python 枚举类，不涉及 Tensor 或设备迁移。

注意：本接口为纯 Python 枚举类型，无浮点 Tensor 输出，
     按 API 语义使用成员名称、值、身份、集成属性与异常断言验证。
"""
import os
import tempfile
import unittest
from enum import Enum

import torch
from torch.distributed.checkpoint.filesystem import FileSystemWriter, SerializationFormat

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
class TestSerializationFormat(TestCase):
    def test_members_lookup_and_identity(self):
        self.assertTrue(issubclass(SerializationFormat, Enum))
        self.assertEqual(list(SerializationFormat), [SerializationFormat.TORCH_SAVE, SerializationFormat.SAFETENSORS])
        for name, val, member in (
            ("TORCH_SAVE", "torch_save", SerializationFormat.TORCH_SAVE),
            ("SAFETENSORS", "safetensors", SerializationFormat.SAFETENSORS),
        ):
            self.assertEqual(member.name, name)
            self.assertEqual(member.value, val)
            self.assertIs(SerializationFormat(val), member)
            self.assertIs(SerializationFormat[name], member)

        self.assertNotEqual(SerializationFormat.TORCH_SAVE, SerializationFormat.SAFETENSORS)
        mapping = {SerializationFormat.TORCH_SAVE: 1, SerializationFormat.SAFETENSORS: 2}
        self.assertEqual(mapping[SerializationFormat.TORCH_SAVE], 1)
        self.assertEqual(mapping[SerializationFormat.SAFETENSORS], 2)

    def test_filesystem_writer_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIs(FileSystemWriter(tmpdir).serialization_format, SerializationFormat.TORCH_SAVE)
            st_writer = FileSystemWriter(tmpdir, serialization_format=SerializationFormat.SAFETENSORS)
            self.assertIs(st_writer.serialization_format, SerializationFormat.SAFETENSORS)

    def test_invalid_lookup_raises(self):
        for bad_val in ("unknown_format", ""):
            with self.assertRaises(ValueError):
                SerializationFormat(bad_val)
        with self.assertRaises(KeyError):
            _ = SerializationFormat["UNKNOWN"]


if __name__ == "__main__":
    run_tests()
