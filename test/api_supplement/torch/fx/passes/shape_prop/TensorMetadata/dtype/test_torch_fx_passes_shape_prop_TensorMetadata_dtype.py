# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.shape_prop.TensorMetadata.dtype 接口功能正确性
API 名称：torch.fx.passes.shape_prop.TensorMetadata.dtype
API 签名：TensorMetadata.dtype: torch.dtype（只读字段）

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景的空形状、空步长或空量化字典 |
| 枚举选项 | 已覆盖：相关场景中的 float32/int64 或连续/通道后置/未指定内存格式 |
| 参数类型 | 已覆盖：torch.Size、dtype、布尔值、stride 元组和量化参数字典 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：字段只读和其他字段保持 |
| 正常传参场景 | 已覆盖：dtype 字段读取、元组往返与替换结果 |
| 异常传参场景 | 已覆盖：记录字段不可赋值，抛出 AttributeError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
"""
import os
import unittest

import torch

RUN_FLAG = os.environ.get("RUN_SUPPLEMENT_API_TESTS") == "1"
SKIP_REASON = "Supplement API tests run only in full regression (set RUN_SUPPLEMENT_API_TESTS=1)"
DEVICE_NAME = None
if RUN_FLAG:
    try:
        import torch_npu  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch_npu":
            raise
    else:
        if torch.npu.is_available():
            DEVICE_NAME = "npu"
    if DEVICE_NAME is None and torch.cuda.is_available():
        DEVICE_NAME = "cuda"

if DEVICE_NAME == "npu":
    from torch_npu.testing.testcase import TestCase, run_tests
else:
    from unittest import TestCase

    def run_tests():
        unittest.main()

from torch.fx.passes.shape_prop import TensorMetadata


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_dtype_field_values_and_tuple_roundtrip(self):
        record = TensorMetadata(torch.Size([2, 3]), torch.float32, False,
                                (3, 1), None, False, {})._replace(dtype=torch.float32)
        self.assertEqual(record.dtype, torch.float32)
        updated = record._replace(dtype=torch.int64)
        self.assertEqual(updated.dtype, torch.int64)
        self.assertEqual(record.dtype, torch.float32)
        self.assertEqual(TensorMetadata(*tuple(updated)).dtype, torch.int64)

    def test_dtype_field_is_read_only_and_replace_preserves_other_fields(self):
        record = TensorMetadata(torch.Size([]), torch.float32, False, (), None, False, {})
        before = record._asdict()
        changed = record._replace(dtype=torch.float16)
        self.assertEqual(changed.dtype, torch.float16)
        for name, value in before.items():
            if name != "dtype":
                self.assertEqual(getattr(changed, name), value)
        with self.assertRaises(AttributeError):
            record.dtype = torch.float16


if __name__ == "__main__":
    run_tests()
