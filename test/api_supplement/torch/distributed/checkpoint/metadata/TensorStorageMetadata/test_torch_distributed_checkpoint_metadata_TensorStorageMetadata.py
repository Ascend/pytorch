# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.metadata.TensorStorageMetadata 接口功能正确性
API 名称：torch.distributed.checkpoint.metadata.TensorStorageMetadata
API 签名：TensorStorageMetadata(properties, size, chunks)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 chunks 列表、单分片、多分片与 0 元素 Tensor | 已覆盖（test_construction_and_boundary_shapes） |
| 枚举选项 | 不同 dtype 与 memory_format 组合 | 已覆盖（test_construction_and_boundary_shapes、test_mutability_and_pickle） |
| 参数类型 | TensorProperties、Size、Chunk 列表 | 已覆盖（test_construction_and_boundary_shapes） |
| 传参与不传参 | 3 个必选参数位置/关键字传参与缺参报错 | 已覆盖（test_construction_and_boundary_shapes、test_missing_args_raises） |
| 等价类/边界值 | 0D/空张量、chunks 变更、replace | 已覆盖（test_construction_and_boundary_shapes、test_mutability_and_pickle） |
| 正常传参场景 | 构造、pickle 往返、Metadata 集成 | 已覆盖（test_construction_and_boundary_shapes、test_mutability_and_pickle） |
| 异常传参场景 | 缺少必选参数抛出 TypeError | 已覆盖（test_missing_args_raises） |
| 混合设备类型 | 不涉及多 Tensor 混合设备计算（从目标设备 Tensor 提取属性） | 未覆盖（纯 Python 元数据类，仅记录张量属性与分片边界） |

未覆盖项及原因：
- 混合设备类型：TensorStorageMetadata 为纯 Python 存储元数据类，不持有设备张量。

注意：本接口为纯 Python 元数据 dataclass，无浮点 Tensor 数值输出，
     按 API 语义使用字段值、序列化往返一致性与异常断言验证。
"""
import dataclasses
import os
import pickle
import unittest

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

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
class TestTensorStorageMetadata(TestCase):
    def test_construction_and_boundary_shapes(self):
        tensor = torch.randn(4, 6, dtype=torch.float32, device=DEVICE_NAME)
        props = TensorProperties.create_from_tensor(tensor)

        empty_chunks_md = TensorStorageMetadata(properties=props, size=tensor.size(), chunks=[])
        self.assertEqual(empty_chunks_md.properties, props)
        self.assertEqual(empty_chunks_md.size, torch.Size([4, 6]))
        self.assertEqual(empty_chunks_md.chunks, [])

        c0 = ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([2, 6]))
        c1 = ChunkStorageMetadata(offsets=torch.Size([2, 0]), sizes=torch.Size([2, 6]))
        sharded_md = TensorStorageMetadata(props, torch.Size([4, 6]), [c0, c1])
        self.assertEqual(sharded_md.chunks, [c0, c1])

        int_props = TensorProperties(dtype=torch.int64)
        scalar_md = TensorStorageMetadata(
            properties=int_props,
            size=torch.Size([]),
            chunks=[ChunkStorageMetadata(offsets=torch.Size([]), sizes=torch.Size([]))],
        )
        self.assertEqual(scalar_md.size.numel(), 1)
        self.assertEqual(len(scalar_md.chunks), 1)

        empty_tensor_md = TensorStorageMetadata(
            properties=int_props,
            size=torch.Size([0, 4]),
            chunks=[ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([0, 4]))],
        )
        self.assertEqual(empty_tensor_md.size.numel(), 0)
        self.assertEqual(empty_tensor_md.chunks[0].sizes, torch.Size([0, 4]))

    def test_mutability_and_pickle(self):
        props = TensorProperties(dtype=torch.float16)
        c0 = ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([4]))
        c1 = ChunkStorageMetadata(offsets=torch.Size([4]), sizes=torch.Size([4]))

        m1 = TensorStorageMetadata(properties=props, size=torch.Size([8]), chunks=[c0])
        m2 = TensorStorageMetadata(properties=props, size=torch.Size([8]), chunks=[c0])
        self.assertEqual(m1, m2)

        m1.chunks.append(c1)
        self.assertNotEqual(m1, m2)
        self.assertEqual(len(m1.chunks), 2)

        m3 = dataclasses.replace(m2, size=torch.Size([16]))
        self.assertEqual(m3.size, torch.Size([16]))
        self.assertEqual(m2.size, torch.Size([8]))

        bf16_props = TensorProperties(dtype=torch.bfloat16, memory_format=torch.channels_last)
        chunk = ChunkStorageMetadata(offsets=torch.Size([0, 0]), sizes=torch.Size([3, 5]))
        tsm = TensorStorageMetadata(properties=bf16_props, size=torch.Size([3, 5]), chunks=[chunk])
        self.assertEqual(pickle.loads(pickle.dumps(tsm)), tsm)

        ckpt_md = Metadata(state_dict_metadata={"layer.weight": tsm})
        self.assertEqual(pickle.loads(pickle.dumps(ckpt_md)).state_dict_metadata["layer.weight"], tsm)

    def test_missing_args_raises(self):
        with self.assertRaises(TypeError):
            TensorStorageMetadata()
        with self.assertRaises(TypeError):
            TensorStorageMetadata(properties=TensorProperties(), size=torch.Size([2]))


if __name__ == "__main__":
    run_tests()
