# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint_transformation.valid_index 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint_transformation.valid_index
API 签名：valid_index(index, dims) -> torch.fx.experimental.migrate_gradual_types.constraint.Constraint

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空维度、空分支或空约束 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：维度变量、静态维度、动态类型或真实回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：左右越界和空维度列表 |
| 正常传参场景 | 已覆盖：Python 正/负索引合法边界 |
| 异常传参场景 | 不适用：有效约束或索引场景 |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 约束场景使用 Z3 的 SAT/UNSAT 作为独立结果断言；不声明数值算子执行或 NPU codegen 覆盖。
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

from torch.fx.experimental.migrate_gradual_types.constraint import DVar, T, F
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import valid_index


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_valid_index_positive_and_negative_valid_indices(self):
        dimensions = [DVar(1), DVar(2), DVar(3)]
        for index in (-3, -1, 0, 2):
            with self.subTest(index=index):
                self.assertIsInstance(valid_index(index, dimensions), T)

    def test_valid_index_out_of_range_and_empty_dimensions(self):
        dimensions = [DVar(1), DVar(2), DVar(3)]
        self.assertIsInstance(valid_index(-4, dimensions), F)
        self.assertIsInstance(valid_index(3, dimensions), F)
        self.assertIsInstance(valid_index(0, []), F)
        self.assertIsInstance(valid_index(-1, []), F)


if __name__ == "__main__":
    run_tests()
