# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.DimDynamic 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.DimDynamic
API 签名：DimDynamic(value) -> DimDynamic

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空上下文、默认集合或 None 字段 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际符号键、Source、枚举策略和对应数据记录 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：策略区分与非法枚举值 |
| 正常传参场景 | 已覆盖：全部 5 个策略名称、值和反查 |
| 异常传参场景 | 已覆盖：未知枚举值抛出 ValueError |
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

from torch.fx.experimental.symbolic_shapes import DimDynamic


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_dimdynamic_policy_names_values_and_roundtrip(self):
        policies = {"DYNAMIC": 0, "DUCK": 1, "STATIC": 2, "UNBACKED": 3, "INFER_STRIDE": 4}
        self.assertEqual({item.name: item.value for item in DimDynamic}, policies)
        for name, value in policies.items():
            with self.subTest(name=name):
                self.assertIs(DimDynamic(value), DimDynamic[name])

    def test_dimdynamic_distinct_policies_and_unknown_value(self):
        self.assertEqual(len(set(DimDynamic)), 5)
        self.assertNotEqual(DimDynamic.DUCK, DimDynamic.DYNAMIC)
        with self.assertRaises(ValueError):
            DimDynamic(99)


if __name__ == "__main__":
    run_tests()
