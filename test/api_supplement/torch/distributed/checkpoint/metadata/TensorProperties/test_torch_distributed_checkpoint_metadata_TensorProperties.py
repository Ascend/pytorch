# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.checkpoint.metadata.TensorProperties 接口功能正确性
API 名称：torch.distributed.checkpoint.metadata.TensorProperties
API 签名：TensorProperties(dtype=..., layout=torch.strided, requires_grad=False, memory_format=..., pin_memory=False)

覆盖维度表：
| 覆盖维度 | 说明 | 覆盖情况 |
|----------|------|----------|
| 空/非空 | 0 元素空 Tensor、0D 标量与多维 Tensor 创建属性 | 已覆盖（test_construction_and_create_from_tensor） |
| 枚举选项 | memory_format 三种值及多种 dtype | 已覆盖（test_construction_and_create_from_tensor、test_pickle_equality_and_replace） |
| 参数类型 | dtype/layout/bool/memory_format 与 Tensor | 已覆盖（test_construction_and_create_from_tensor） |
| 传参与不传参 | 默认无参构造与显式全参数构造 | 已覆盖（test_construction_and_create_from_tensor） |
| 等价类/边界值 | 相等性/不等性、dataclasses.replace、pickle 往返 | 已覆盖（test_pickle_equality_and_replace） |
| 正常传参场景 | 构造、create_from_tensor 及 __getstate__/__setstate__ | 已覆盖（全部正常用例） |
| 异常传参场景 | 非法 memory_format 序列化或非法编码反序列化报错 | 已覆盖（test_invalid_memory_format_raises） |
| 混合设备类型 | 目标设备 Tensor 与 CPU Tensor 分别构造属性 | 已覆盖（test_construction_and_create_from_tensor） |

未覆盖项及原因：
- 无

注意：本接口为纯 Python 元数据 dataclass，无浮点 Tensor 数值输出，
     按 API 语义使用字段值、序列化往返一致性与异常断言验证。
"""
import dataclasses
import os
import pickle
import unittest

import torch
from torch.distributed.checkpoint.metadata import TensorProperties

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
class TestTensorProperties(TestCase):
    def test_construction_and_create_from_tensor(self):
        default_props = TensorProperties()
        self.assertEqual(default_props.dtype, torch.get_default_dtype())
        self.assertEqual(default_props.layout, torch.strided)
        self.assertFalse(default_props.requires_grad)
        self.assertEqual(default_props.memory_format, torch.contiguous_format)
        self.assertFalse(default_props.pin_memory)

        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool):
            props = TensorProperties(
                dtype=dtype,
                layout=torch.strided,
                requires_grad=(dtype == torch.float32),
                memory_format=torch.channels_last,
                pin_memory=True,
            )
            self.assertEqual(props.dtype, dtype)
            self.assertEqual(props.requires_grad, dtype == torch.float32)
            self.assertEqual(props.memory_format, torch.channels_last)
            self.assertTrue(props.pin_memory)

        dev_tensor = torch.randn(2, 3, dtype=torch.float32, device=DEVICE_NAME, requires_grad=True)
        dev_props = TensorProperties.create_from_tensor(dev_tensor)
        self.assertEqual(dev_props.dtype, torch.float32)
        self.assertTrue(dev_props.requires_grad)
        self.assertEqual(dev_props.memory_format, torch.contiguous_format)
        self.assertEqual(dev_props.pin_memory, dev_tensor.is_pinned())

        empty_props = TensorProperties.create_from_tensor(torch.empty(0, 4, dtype=torch.int64, device=DEVICE_NAME))
        self.assertEqual(empty_props.dtype, torch.int64)
        self.assertFalse(empty_props.requires_grad)

        cpu_props = TensorProperties.create_from_tensor(torch.tensor(3.0, dtype=torch.float64, device="cpu"))
        self.assertEqual(cpu_props.dtype, torch.float64)
        self.assertFalse(cpu_props.requires_grad)
        self.assertFalse(cpu_props.pin_memory)

    def test_pickle_equality_and_replace(self):
        for mem_fmt in (torch.contiguous_format, torch.channels_last, torch.preserve_format):
            props = TensorProperties(
                dtype=torch.float16,
                layout=torch.strided,
                requires_grad=True,
                memory_format=mem_fmt,
                pin_memory=True,
            )
            restored = pickle.loads(pickle.dumps(props))
            self.assertEqual(restored, props)
            self.assertEqual(restored.memory_format, mem_fmt)

        p1 = TensorProperties(dtype=torch.float32, requires_grad=False)
        p2 = TensorProperties(dtype=torch.float32, requires_grad=False)
        p3 = dataclasses.replace(p1, requires_grad=True)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertTrue(p3.requires_grad)

    def test_invalid_memory_format_raises(self):
        with self.assertRaises(RuntimeError):
            TensorProperties(memory_format=torch.channels_last_3d).__getstate__()
        with self.assertRaises(RuntimeError):
            TensorProperties().__setstate__((torch.float32, torch.strided, False, "invalid_encoding", False))


if __name__ == "__main__":
    run_tests()
