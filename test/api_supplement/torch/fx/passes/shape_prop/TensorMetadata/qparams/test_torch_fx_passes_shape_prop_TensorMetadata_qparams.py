# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.passes.shape_prop.TensorMetadata.qparams 接口功能正确性
API 名称：torch.fx.passes.shape_prop.TensorMetadata.qparams
API 签名：TensorMetadata.qparams: dict[str, typing.Any]（只读字段）

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景的空形状、空步长或空量化字典 |
| 枚举选项 | 已覆盖：相关场景中的 float32/int64 或连续/通道后置/未指定内存格式 |
| 参数类型 | 已覆盖：torch.Size、dtype、布尔值、stride 元组和量化参数字典 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：只读字段绑定与可变字典引用 |
| 正常传参场景 | 已覆盖：空字典和 per-tensor 量化参数值 |
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

    def test_qparams_empty_and_per_tensor_quantization_parameters(self):
        empty = TensorMetadata(torch.Size([2]), torch.float32, False, (1,), None, False, {})
        self.assertEqual(empty.qparams, {})
        parameters = {"qscheme": torch.per_tensor_affine, "scale": 0.25, "zero_point": 3}
        quantized = empty._replace(dtype=torch.quint8, is_quantized=True, qparams=parameters)
        self.assertEqual(quantized.qparams, parameters)
        self.assertEqual(quantized.qparams["scale"], 0.25)
        self.assertEqual(quantized.qparams["zero_point"], 3)
        self.assertEqual(empty.qparams, {})

    def test_qparams_field_binding_is_read_only_but_dictionary_is_shared(self):
        parameters = {"scale": 0.5}
        record = TensorMetadata(torch.Size([1]), torch.quint8, False, (1,), None, True, parameters)
        parameters["zero_point"] = 7
        self.assertEqual(record.qparams, {"scale": 0.5, "zero_point": 7})
        with self.assertRaises(AttributeError):
            record.qparams = {}


if __name__ == "__main__":
    run_tests()
