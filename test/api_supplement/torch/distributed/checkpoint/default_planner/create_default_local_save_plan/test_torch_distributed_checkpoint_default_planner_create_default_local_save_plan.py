# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.default_planner.create_default_local_save_plan 接口功能正确性
API 名称：torch.distributed.checkpoint.default_planner.create_default_local_save_plan
API 签名：create_default_local_save_plan(state_dict, is_coordinator) -> SavePlan

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 空 dict、0 元素与非空 Tensor/非 Tensor | 已覆盖（test_empty_and_device_tensors、test_mixed_devices_and_non_tensor_objects） |
| 枚举选项 | is_coordinator、TENSOR/BYTE_IO | 已覆盖（test_empty_and_device_tensors、test_mixed_devices_and_non_tensor_objects） |
| 参数类型 | 多 dtype Tensor 与非 Tensor 对象 | 已覆盖（test_empty_and_device_tensors、test_mixed_devices_and_non_tensor_objects） |
| 传参与不传参 | 两个必选参数显式传参、缺参报错 | 已覆盖（test_empty_and_device_tensors、test_missing_args_raises） |
| 等价类/边界值 | 0D 标量、0 元素空 Tensor、1D/2D、requires_grad=True | 已覆盖（test_empty_and_device_tensors） |
| 正常传参场景 | 多类型混合 state_dict 生成完整 SavePlan | 已覆盖（test_mixed_devices_and_non_tensor_objects） |
| 异常传参场景 | 缺少必选参数或传入非 dict 对象报错 | 已覆盖（test_missing_args_raises） |
| 混合设备类型 | 同一 state_dict 同时包含目标设备与 CPU Tensor | 已覆盖（test_mixed_devices_and_non_tensor_objects） |

未覆盖项及原因：
- 无

注意：本接口为纯 Python 本地保存计划生成函数，无浮点 Tensor 数值输出，
     按 API 语义使用返回 SavePlan 的 WriteItem 类型、尺寸、属性及异常断言验证。
"""
import os
import unittest

import torch
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType

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
class TestCreateDefaultLocalSavePlan(TestCase):
    def test_empty_and_device_tensors(self):
        for is_coord in (True, False):
            empty_plan = create_default_local_save_plan({}, is_coordinator=is_coord)
            self.assertIsInstance(empty_plan, SavePlan)
            self.assertEqual(empty_plan.items, [])
            self.assertTrue(empty_plan.usable)

        state_dict = {
            "weight": torch.randn(4, 3, dtype=torch.float32, device=DEVICE_NAME, requires_grad=True),
            "scalar": torch.tensor(7, dtype=torch.int64, device=DEVICE_NAME),
            "empty": torch.empty(0, 5, dtype=torch.float32, device=DEVICE_NAME),
            "mask": torch.tensor([True, False], dtype=torch.bool, device=DEVICE_NAME),
        }
        plan = create_default_local_save_plan(state_dict, is_coordinator=True)
        items_by_fqn = {item.index.fqn: item for item in plan.items}
        self.assertEqual(set(items_by_fqn.keys()), set(state_dict.keys()))

        w_item = items_by_fqn["weight"]
        self.assertEqual(w_item.type, WriteItemType.TENSOR)
        self.assertEqual(w_item.tensor_data.size, torch.Size([4, 3]))
        self.assertEqual(w_item.tensor_data.chunk.offsets, torch.Size([0, 0]))
        self.assertEqual(w_item.tensor_data.chunk.sizes, torch.Size([4, 3]))
        self.assertEqual(w_item.tensor_data.properties.dtype, torch.float32)
        self.assertTrue(w_item.tensor_data.properties.requires_grad)

        self.assertEqual(items_by_fqn["scalar"].tensor_data.size, torch.Size([]))
        self.assertEqual(items_by_fqn["scalar"].tensor_data.properties.dtype, torch.int64)
        self.assertEqual(items_by_fqn["empty"].tensor_data.size, torch.Size([0, 5]))
        self.assertEqual(items_by_fqn["empty"].tensor_storage_size(), 0)
        self.assertEqual(items_by_fqn["mask"].tensor_data.properties.dtype, torch.bool)

    def test_mixed_devices_and_non_tensor_objects(self):
        state_dict = {
            "dev_t": torch.ones(2, 2, device=DEVICE_NAME),
            "cpu_t": torch.zeros(3, device="cpu"),
            "epoch": 10,
            "config": {"lr": 1e-3, "betas": [0.9, 0.999]},
        }
        expected = {
            "dev_t": WriteItemType.TENSOR,
            "cpu_t": WriteItemType.TENSOR,
            "epoch": WriteItemType.BYTE_IO,
            "config": WriteItemType.BYTE_IO,
        }
        for is_coord in (True, False):
            plan = create_default_local_save_plan(state_dict, is_coordinator=is_coord)
            self.assertEqual({item.index.fqn: item.type for item in plan.items}, expected)
            for item in plan.items:
                if item.type == WriteItemType.BYTE_IO:
                    self.assertIsNone(item.tensor_data)
                    self.assertIsNone(item.tensor_storage_size())

    def test_missing_args_raises(self):
        with self.assertRaises(TypeError):
            create_default_local_save_plan()
        with self.assertRaises(TypeError):
            create_default_local_save_plan({})
        with self.assertRaises(AttributeError):
            create_default_local_save_plan(["not_a_dict"], is_coordinator=True)


if __name__ == "__main__":
    run_tests()
