# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.default_planner.create_default_global_load_plan 接口功能正确性
API 名称：torch.distributed.checkpoint.default_planner.create_default_global_load_plan
API 签名：create_default_global_load_plan(all_plans) -> list[LoadPlan]

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空列表、含空 LoadPlan 与含多项 ReadItem 的计划列表 | 已覆盖（test_empty_and_multi_rank_plans） |
| 枚举选项 | ReadItem 的 TENSOR/BYTE_IO 类型 | 已覆盖（test_empty_and_multi_rank_plans、test_integration_with_local_load_plan） |
| 参数类型 | list[LoadPlan] 及 storage_data/planner_data | 已覆盖（test_empty_and_multi_rank_plans） |
| 传参与不传参 | 必选参数显式传入与缺参报错 | 已覆盖（test_empty_and_multi_rank_plans、test_missing_arg_raises） |
| 等价类/边界值 | 0 个、1 个、多 rank 计划列表，保留原顺序与内容 | 已覆盖（test_empty_and_multi_rank_plans） |
| 正常传参场景 | 独立构造与本地加载计划联动 | 已覆盖（test_empty_and_multi_rank_plans、test_integration_with_local_load_plan） |
| 异常传参场景 | 缺少必选参数抛出 TypeError | 已覆盖（test_missing_arg_raises） |
| 混合设备类型 | CPU 与目标设备 Tensor 生成的本地加载计划整合 | 已覆盖（test_integration_with_local_load_plan） |

未覆盖项及原因：
- 无

注意：本接口为纯 Python 计划编排函数，无浮点 Tensor 输出，
     按 API 语义使用返回计划结构、元素顺序、元数据字段一致性与异常断言验证。
"""
import os
import unittest

import torch
from torch.distributed.checkpoint.default_planner import (
    create_default_global_load_plan,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_local_save_plan,
)
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, ReadItem

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


def _make_read_item(fqn, item_type, offset, length):
    idx = MetadataIndex(fqn=fqn, offset=offset)
    return ReadItem(
        type=item_type,
        dest_index=idx,
        dest_offsets=torch.Size(offset),
        storage_index=idx,
        storage_offsets=torch.Size(offset),
        lengths=torch.Size(length),
    )


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestCreateDefaultGlobalLoadPlan(TestCase):
    def test_empty_and_multi_rank_plans(self):
        self.assertEqual(create_default_global_load_plan([]), [])
        empty_plan = LoadPlan(items=[])
        empty_res = create_default_global_load_plan([empty_plan])
        self.assertEqual(len(empty_res), 1)
        self.assertEqual(empty_res[0].items, [])
        self.assertIs(empty_res[0], empty_plan)

        item_tensor_r0 = _make_read_item("layer.weight", LoadItemType.TENSOR, [0, 0], [2, 4])
        item_bytes_r0 = _make_read_item("meta.step", LoadItemType.BYTE_IO, [], [])
        item_tensor_r1 = _make_read_item("layer.weight", LoadItemType.TENSOR, [2, 0], [2, 4])

        plan_r0 = LoadPlan(
            items=[item_tensor_r0, item_bytes_r0], storage_data={"rank": 0}, planner_data={"mapping": "r0"}
        )
        plan_r1 = LoadPlan(items=[item_tensor_r1], storage_data={"rank": 1}, planner_data={"mapping": "r1"})

        global_plans = create_default_global_load_plan([plan_r0, plan_r1])
        self.assertEqual(len(global_plans), 2)
        self.assertEqual(global_plans[0].items, [item_tensor_r0, item_bytes_r0])
        self.assertEqual(global_plans[1].items, [item_tensor_r1])
        self.assertEqual(global_plans[0].storage_data, {"rank": 0})
        self.assertEqual(global_plans[1].planner_data, {"mapping": "r1"})

    def test_integration_with_local_load_plan(self):
        state_dict = {
            "dev_weight": torch.randn(4, 3, device=DEVICE_NAME),
            "cpu_bias": torch.randn(3, device="cpu"),
            "step": 42,
        }
        local_save = create_default_local_save_plan(state_dict, is_coordinator=True)
        _, metadata = create_default_global_save_plan([local_save])

        local_loads = [create_default_local_load_plan(state_dict, metadata, strict=True) for _ in range(2)]
        global_plans = create_default_global_load_plan(local_loads)
        self.assertEqual(len(global_plans), 2)
        for plan in global_plans:
            self.assertEqual({item.dest_index.fqn for item in plan.items}, {"dev_weight", "cpu_bias", "step"})

    def test_missing_arg_raises(self):
        with self.assertRaises(TypeError):
            create_default_global_load_plan()


if __name__ == "__main__":
    run_tests()
