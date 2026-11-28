# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.validator.bisect 接口功能正确性
API 名称：torch.fx.experimental.validator.bisect
API 签名：bisect(shape_env) -> None

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空状态、约束或记录 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际 ShapeEnv、SymPy 表达式及对应默认/显式参数 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：关闭二分时真实验证错误传播及配置恢复 |
| 正常传参场景 | 已覆盖：真实翻译验证成功路径 |
| 异常传参场景 | 已覆盖：Z3 反例产生的 ValidationException |
| 混合设备类型 | 不适用：本批为设备无关的图结构、类型或辅助状态语义 |

未覆盖项及原因：
- 图执行方式：本文件直接验证图模式辅助接口，不声明 NPU Inductor codegen 覆盖。
- eager 对照：使用接口定义的结构、状态或确定值；不构造无意义的 CPU/NPU 自比对。
- autograd：本接口场景不计算梯度。
- RNG：本文件使用确定输入，不依赖随机流。
- 参数类型：未覆盖带多个失败节点的事件二分搜索与最早节点定位。
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

import sympy
import z3
from torch.fx.experimental import _config as config
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.experimental.validator import ValidationException, bisect


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_bisect_successful_translation_returns_without_modifying_events(self):
        with config.patch(translation_validation=True):
            env = ShapeEnv(should_record_events=True, tracked_fakes=[])
            self.addCleanup(env.cleanup)
            events = tuple(env.events)
            self.assertIsNone(bisect(env))
            self.assertEqual(tuple(env.events), events)

    def test_bisect_validation_failure_is_preserved_when_bisection_is_disabled(self):
        original = config.translation_validation_no_bisect
        with config.patch(translation_validation=True, translation_validation_no_bisect=True):
            env = ShapeEnv(should_record_events=False, tracked_fakes=[])
            self.addCleanup(env.cleanup)
            env.validator.add_source_expr(z3.BoolVal(False))
            env.validator.add_target_expr(sympy.true)
            with self.assertRaises(ValidationException):
                bisect(env)
        self.assertEqual(config.translation_validation_no_bisect, original)


if __name__ == "__main__":
    run_tests()
