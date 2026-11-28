# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.meta_tracer.embedding_override 接口功能正确性
API 名称：torch.fx.experimental.meta_tracer.embedding_override
API 签名：embedding_override(self, input) -> torch.Tensor

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的零尺寸或标量输入 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：meta Tensor、对应 nn.Module 或真实 FX Proxy |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零尺寸索引和标量索引 |
| 正常传参场景 | 已覆盖：输入索引形状追加 embedding_dim |
| 异常传参场景 | 不适用：有效 meta 分析输入 |
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

from torch.fx.experimental.meta_tracer import embedding_override


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_embedding_override_embedding_output_appends_embedding_dimension(self):
        module = torch.nn.Embedding(11, 4, device="meta")
        indices = torch.empty(2, 3, dtype=torch.int64, device="meta")
        result = embedding_override(module, indices)
        self.assertEqual(result.shape, torch.Size([2, 3, 4]))
        self.assertEqual(result.device.type, "meta")
        self.assertEqual(indices.shape, torch.Size([2, 3]))

    def test_embedding_override_empty_and_scalar_index_shapes(self):
        module = torch.nn.Embedding(11, 5, padding_idx=0, device="meta")
        empty = embedding_override(module, torch.empty(0, 2, dtype=torch.int64, device="meta"))
        scalar = embedding_override(module, torch.empty((), dtype=torch.int64, device="meta"))
        self.assertEqual(empty.shape, torch.Size([0, 2, 5]))
        self.assertEqual(scalar.shape, torch.Size([5]))
        self.assertEqual(empty.numel(), 0)


if __name__ == "__main__":
    run_tests()
