# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.default_planner.create_default_global_save_plan 接口功能正确性
API 名称：torch.distributed.checkpoint.default_planner.create_default_global_save_plan
API 签名：create_default_global_save_plan(all_plans, rewrite_index_hints=True) -> tuple[list[SavePlan], Metadata]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表、空计划、Tensor/Bytes/Shard | 已覆盖（test_empty_and_tensor_bytes_plans、test_sharded_chunks_and_index_hints） |
| 枚举选项 | TENSOR / BYTE_IO / SHARD | 已覆盖（test_empty_and_tensor_bytes_plans、test_sharded_chunks_and_index_hints） |
| 参数类型 | SavePlan 列表与 rewrite_index_hints | 已覆盖（test_empty_and_tensor_bytes_plans、test_sharded_chunks_and_index_hints） |
| 传参与不传参 | rewrite_index_hints 默认与显式 True/False | 已覆盖（test_sharded_chunks_and_index_hints） |
| 等价类/边界值 | 单分片、多分片合并、保留 storage/planner_data | 已覆盖（test_sharded_chunks_and_index_hints） |
| 正常传参场景 | 设备 Tensor 构造本地计划并生成全局元数据 | 已覆盖（test_empty_and_tensor_bytes_plans、test_sharded_chunks_and_index_hints） |
| 异常传参场景 | 非 SHARD 重复 fqn、缺失 tensor_data 或 chunk、缺参 | 已覆盖（test_error_paths） |
| 混合设备类型 | 目标设备 Tensor 与 CPU Tensor 混合 state_dict 生成计划 | 已覆盖（test_empty_and_tensor_bytes_plans） |

未覆盖项及原因：
- 无

注意：本接口为纯 Python 全局保存计划生成函数，无浮点 Tensor 输出，
     按 API 语义使用返回的 SavePlan 列表、Metadata 元数据及异常断言验证。
"""
import os
import unittest

import torch
from torch.distributed.checkpoint.default_planner import (
    create_default_global_save_plan,
    create_default_local_save_plan,
)
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import SavePlan, TensorWriteData, WriteItem, WriteItemType

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


def _make_shard_write_item(fqn, offset, chunk_size, full_size, index_hint=None):
    chunk = ChunkStorageMetadata(offsets=torch.Size(offset), sizes=torch.Size(chunk_size))
    return WriteItem(
        index=MetadataIndex(fqn=fqn, offset=offset, index=index_hint),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=chunk, properties=TensorProperties(dtype=torch.float32), size=torch.Size(full_size)
        ),
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateDefaultGlobalSavePlan(TestCase):
    def test_empty_and_tensor_bytes_plans(self):
        plans, md = create_default_global_save_plan([])
        self.assertEqual(plans, [])
        self.assertIsInstance(md, Metadata)
        self.assertEqual(md.state_dict_metadata, {})

        plans_single, md_single = create_default_global_save_plan([SavePlan(items=[])])
        self.assertEqual(len(plans_single), 1)
        self.assertEqual(plans_single[0].items, [])
        self.assertEqual(md_single.state_dict_metadata, {})

        state_dict = {
            "dev_w": torch.randn(3, 4, dtype=torch.float32, device=DEVICE_NAME),
            "cpu_b": torch.zeros(4, dtype=torch.int64, device="cpu"),
            "meta_cfg": {"lr": 0.01},
        }
        local_plan = create_default_local_save_plan(state_dict, is_coordinator=True)
        global_plans, md = create_default_global_save_plan([local_plan])

        self.assertEqual(len(global_plans), 1)
        self.assertEqual(set(md.state_dict_metadata.keys()), {"dev_w", "cpu_b", "meta_cfg"})

        dev_md = md.state_dict_metadata["dev_w"]
        self.assertIsInstance(dev_md, TensorStorageMetadata)
        self.assertEqual(dev_md.size, torch.Size([3, 4]))
        self.assertEqual(dev_md.properties.dtype, torch.float32)
        self.assertEqual(len(dev_md.chunks), 1)
        self.assertEqual(dev_md.chunks[0].offsets, torch.Size([0, 0]))
        self.assertEqual(dev_md.chunks[0].sizes, torch.Size([3, 4]))

        cpu_md = md.state_dict_metadata["cpu_b"]
        self.assertIsInstance(cpu_md, TensorStorageMetadata)
        self.assertEqual(cpu_md.size, torch.Size([4]))
        self.assertEqual(cpu_md.properties.dtype, torch.int64)
        self.assertIsInstance(md.state_dict_metadata["meta_cfg"], BytesStorageMetadata)

    def test_sharded_chunks_and_index_hints(self):
        shard0 = _make_shard_write_item("sharded.w", [0, 0], [2, 4], [4, 4], index_hint=99)
        shard1 = _make_shard_write_item("sharded.w", [2, 0], [2, 4], [4, 4], index_hint=42)
        input_plans = [
            SavePlan(items=[shard0], storage_data="s_data", planner_data={"k": "v"}, usable=False),
            SavePlan(items=[shard1]),
        ]

        plans_rewritten, md_rewritten = create_default_global_save_plan(input_plans)
        self.assertEqual([p.items[0].index.index for p in plans_rewritten], [0, 1])
        self.assertEqual(plans_rewritten[0].storage_data, "s_data")
        self.assertEqual(plans_rewritten[0].planner_data, {"k": "v"})
        self.assertFalse(plans_rewritten[0].usable)

        tensor_md = md_rewritten.state_dict_metadata["sharded.w"]
        self.assertEqual(tensor_md.size, torch.Size([4, 4]))
        self.assertEqual([c.offsets for c in tensor_md.chunks], [torch.Size([0, 0]), torch.Size([2, 0])])

        plans_preserved, md_preserved = create_default_global_save_plan(input_plans, rewrite_index_hints=False)
        self.assertEqual([p.items[0].index.index for p in plans_preserved], [99, 42])
        self.assertEqual(len(md_preserved.state_dict_metadata["sharded.w"].chunks), 2)

    def test_error_paths(self):
        dup_item = WriteItem(index=MetadataIndex("dup_bytes"), type=WriteItemType.BYTE_IO)
        no_tensor_data = WriteItem(index=MetadataIndex("bad_tensor"), type=WriteItemType.TENSOR, tensor_data=None)
        no_chunk = WriteItem(
            index=MetadataIndex("no_chunk"),
            type=WriteItemType.TENSOR,
            tensor_data=TensorWriteData(chunk=None, properties=TensorProperties(), size=torch.Size([2])),
        )
        bad_cases = (
            [SavePlan([dup_item]), SavePlan([dup_item])],
            [SavePlan([no_tensor_data])],
            [SavePlan([no_chunk])],
        )
        for bad_plans in bad_cases:
            with self.assertRaises(AssertionError):
                create_default_global_save_plan(bad_plans)
        with self.assertRaises(TypeError):
            create_default_global_save_plan()


if __name__ == "__main__":
    run_tests()
