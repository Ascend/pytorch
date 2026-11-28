# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.migrate_gradual_types.constraint.is_dim 接口功能正确性
API 名称：torch.fx.experimental.migrate_gradual_types.constraint.is_dim
API 签名：is_dim(d) -> bool

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：相关场景中的空维度、空分支或空约束 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：维度变量、静态维度、动态类型或真实回调 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：零/负维度值及 Tensor 类型与其他对象排除 |
| 正常传参场景 | 已覆盖：int、Dyn 和维度变量 |
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

from torch.fx.tensor_type import Dyn, TensorType
from torch.fx.experimental.migrate_gradual_types.constraint import DVar, TVar, is_dim


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_is_dim_static_dynamic_and_variable_dimensions(self):
        for dimension in (0, 1, -1, 7, Dyn, DVar(3)):
            with self.subTest(dimension=dimension):
                self.assertTrue(is_dim(dimension))

    def test_is_dim_tensor_types_and_non_dimension_objects(self):
        self.assertFalse(is_dim(TVar(1)))
        self.assertFalse(is_dim(TensorType([2, 3])))
        self.assertFalse(is_dim(2.5))
        self.assertFalse(is_dim(None))


if __name__ == "__main__":
    run_tests()
