# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.shape_prop.TensorMetadata 接口功能正确性
API 名称：torch.fx.passes.shape_prop.TensorMetadata
API 签名：TensorMetadata(shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 相关场景的空形状、空步长或空量化字典 |
| 枚举选项 | None、contiguous_format、channels_last、channels_last_3d、preserve_format；量化标志两值 |
| 参数类型 | torch.Size、浮点/整数/bool/complex/量化 dtype、bool、stride、memory_format 和 qparams |
| 传参与不传参 | 不适用：七个字段均必传；位置/关键字两种构造 |
| 等价类/边界值 | 标量、零尺寸、量化字典、字段顺序、引用身份和独立替换 |
| 正常传参场景 | 位置/关键字构造及全部字段顺序 |
| 异常传参场景 | 记录字段不可赋值，抛出 AttributeError |
| 混合设备类型 | 不适用：本文件为设备无关的图结构、类型或辅助状态语义 |

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

    def test_tensormetadata_positional_and_keyword_construction(self):
        positional = TensorMetadata(torch.Size([2, 3]), torch.float32, True,
                                    (3, 1), torch.contiguous_format, False, {})
        named = TensorMetadata(shape=torch.Size([2, 3]), dtype=torch.float32,
                               requires_grad=True, stride=(3, 1),
                               memory_format=torch.contiguous_format, is_quantized=False, qparams={})
        self.assertEqual(positional, named)
        self.assertEqual(tuple(positional), (torch.Size([2, 3]), torch.float32, True,
                                             (3, 1), torch.contiguous_format, False, {}))

    def test_tensormetadata_scalar_record_and_non_mutating_replace(self):
        original = TensorMetadata(torch.Size([]), torch.float32, False, (), None, False, {})
        updated = original._replace(shape=torch.Size([0, 4]), stride=(4, 1), dtype=torch.int64)
        self.assertEqual((original.shape, original.stride, original.dtype),
                         (torch.Size([]), (), torch.float32))
        self.assertEqual((updated.shape, updated.stride, updated.dtype),
                         (torch.Size([0, 4]), (4, 1), torch.int64))
        with self.assertRaises(AttributeError):
            original.shape = torch.Size([9])


    def test_parameter_quantized_metadata_and_all_memory_format_values(self):
        for memory_format in (None, torch.contiguous_format, torch.channels_last,
                              torch.channels_last_3d, torch.preserve_format):
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                          torch.int32, torch.int64, torch.bool, torch.complex64, torch.qint8, torch.quint8):
                parameters = {"qscheme": torch.per_tensor_affine, "scale": 0.25, "zero_point": 3}
                quantized = dtype in (torch.qint8, torch.quint8)
                record = TensorMetadata(torch.Size([0, 3]), dtype, False, (3, 1), memory_format,
                                        quantized, parameters if quantized else {})
                self.assertIs(record.dtype, dtype)
                self.assertIs(record.memory_format, memory_format)
                self.assertIs(record.is_quantized, quantized)
                self.assertEqual(record.shape, torch.Size([0, 3]))
                self.assertEqual(record.stride, (3, 1))
                if quantized:
                    self.assertIs(record.qparams, parameters)
                    self.assertEqual(record.qparams["zero_point"], 3)
                else:
                    self.assertEqual(record.qparams, {})


if __name__ == "__main__":
    run_tests()
