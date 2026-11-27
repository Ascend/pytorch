# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.planner.BytesIOWriteData 接口功能正确性
API 名称：torch.distributed.checkpoint.planner.BytesIOWriteData
API 签名：BytesIOWriteData(nbytes)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | nbytes=0 边界值与正整数值 | 已覆盖（test_construction_and_write_item） |
| 枚举选项 | 不涉及枚举参数 | 未覆盖（dataclass 仅含整型字段 nbytes） |
| 参数类型 | 位置与关键字 int 参数 | 已覆盖（test_construction_and_write_item） |
| 传参与不传参 | 必选参数显式传入与缺参报错 | 已覆盖（test_construction_and_write_item、test_error_paths） |
| 等价类/边界值 | 0/1/大整数、不可变、哈希与 pickle | 已覆盖（test_construction_and_write_item、test_frozen_equality_hashing_and_pickle） |
| 正常传参场景 | 独立构造及嵌入 WriteItem(BYTE_IO) 使用 | 已覆盖（test_construction_and_write_item） |
| 异常传参场景 | 不可变属性赋值抛 FrozenInstanceError、缺参抛 TypeError | 已覆盖（test_error_paths） |
| 混合设备类型 | 不涉及多 Tensor 混合设备输入 | 未覆盖（纯 Python 字节计数 dataclass，不接收 Tensor） |

未覆盖项及原因：
- 枚举选项：BytesIOWriteData 仅包含 nbytes 整型字段，无枚举参数。
- 混合设备类型：纯 Python 数据结构，不涉及 Tensor 或设备迁移。

注意：本接口为纯 Python frozen dataclass，无浮点 Tensor 输出，
     按 API 语义使用字段值、不可变性、哈希、序列化往返与异常断言验证。
"""
import dataclasses
import os
import pickle
import unittest

import torch
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import BytesIOWriteData, WriteItem, WriteItemType

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
class TestBytesIOWriteData(TestCase):
    def test_construction_and_write_item(self):
        d_zero = BytesIOWriteData(0)
        d_one = BytesIOWriteData(nbytes=1)
        d_large = BytesIOWriteData(10**9)
        self.assertEqual(d_zero.nbytes, 0)
        self.assertEqual(d_one.nbytes, 1)
        self.assertEqual(d_large.nbytes, 10**9)
        self.assertTrue(dataclasses.is_dataclass(d_zero))

        bio_data = BytesIOWriteData(nbytes=1024)
        item = WriteItem(
            index=MetadataIndex("optimizer.param_groups"), type=WriteItemType.BYTE_IO, bytes_io_data=bio_data
        )
        self.assertEqual(item.bytes_io_data, bio_data)
        self.assertEqual(item.bytes_io_data.nbytes, 1024)
        self.assertIsNone(item.tensor_storage_size())

    def test_frozen_equality_hashing_and_pickle(self):
        d1 = BytesIOWriteData(256)
        d2 = BytesIOWriteData(nbytes=256)
        d3 = dataclasses.replace(d1, nbytes=512)
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)
        self.assertEqual(hash(d1), hash(d2))
        self.assertEqual({d1: "a", d3: "b"}[d2], "a")

        restored = pickle.loads(pickle.dumps(d1))
        self.assertEqual(restored, d1)
        self.assertEqual(restored.nbytes, 256)

    def test_error_paths(self):
        data = BytesIOWriteData(64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data.nbytes = 128
        with self.assertRaises(TypeError):
            BytesIOWriteData()


if __name__ == "__main__":
    run_tests()
