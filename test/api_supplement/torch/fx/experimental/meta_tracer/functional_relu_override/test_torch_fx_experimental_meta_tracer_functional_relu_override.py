# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.meta_tracer.functional_relu_override 接口功能正确性
API 名称：torch.fx.experimental.meta_tracer.functional_relu_override
API 签名：functional_relu_override(x, inplace=False) -> torch.Tensor

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的零尺寸或标量输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：meta Tensor、对应 nn.Module 或真实 FX Proxy |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零尺寸、显式非就地和禁止就地 |
| 正常传参场景 | 已覆盖：meta 形状、dtype、requires_grad 和原对象保持 |
| 异常传参场景 | 已覆盖：inplace=True 抛出 AssertionError |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 该接口用于 meta 分析；返回原输入的 override 仅验证元数据契约，不声明对应算子的数值正确性。
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

from torch.fx.experimental.meta_tracer import functional_relu_override


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_functional_relu_override_preserves_meta_shape_dtype_and_grad_metadata(self):
        value = torch.empty(2, 3, dtype=torch.float16, device="meta", requires_grad=True)
        result = functional_relu_override(value)
        self.assertIs(result, value)
        self.assertEqual(result.shape, torch.Size([2, 3]))
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(result.requires_grad)

    def test_functional_relu_override_empty_explicit_non_inplace_and_rejected_inplace(self):
        value = torch.empty(0, 3, device="meta")
        self.assertIs(functional_relu_override(value, inplace=False), value)
        with self.assertRaisesRegex(AssertionError, "inplace"):
            functional_relu_override(value, inplace=True)


if __name__ == "__main__":
    run_tests()
