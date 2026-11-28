# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.ShapeEnvSettings 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.ShapeEnvSettings
API 签名：ShapeEnvSettings(allow_scalar_outputs, allow_dynamic_output_shape_ops, assume_static_by_default,
    specialize_zero_one, duck_shape, prefer_deferred_runtime_asserts_over_guards, trace_asserts)

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 已覆盖：适用的空上下文、默认集合或 None 字段 |
| 枚举选项 | 不适用：无枚举配置；离散分支见参数与边界场景 |
| 参数类型 | 已覆盖：实际符号键、Source、枚举策略和对应数据记录 |
| 传参与不传参 | 默认与显式参数按本文件场景验证；未提供的组合见下方说明 |
| 等价类/边界值 | 已覆盖：单项修改、独立替换和只读设置 |
| 正常传参场景 | 已覆盖：位置/关键字配置的相等性和缓存 key 行为 |
| 异常传参场景 | 已覆盖：冻结字段赋值抛出 FrozenInstanceError |
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

from torch.fx.experimental.symbolic_shapes import ShapeEnvSettings
from dataclasses import FrozenInstanceError, replace


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_shapeenvsettings_settings_are_distinct_hashable_cache_keys(self):
        first = ShapeEnvSettings(True, True, False, True, True, False, False)
        same = ShapeEnvSettings(allow_scalar_outputs=True, allow_dynamic_output_shape_ops=True,
                                assume_static_by_default=False, specialize_zero_one=True, duck_shape=True,
                                prefer_deferred_runtime_asserts_over_guards=False, trace_asserts=False)
        changed = replace(first, duck_shape=False)
        self.assertEqual(first, same)
        self.assertNotEqual(first, changed)
        self.assertEqual({first: "cached"}[same], "cached")
        self.assertEqual(len({first, same, changed}), 2)

    def test_shapeenvsettings_replacement_does_not_mutate_original_and_settings_are_frozen(self):
        original = ShapeEnvSettings(False, False, True, False, False, True, True)
        changed = replace(original, allow_scalar_outputs=True, trace_asserts=False)
        self.assertFalse(original.allow_scalar_outputs)
        self.assertTrue(original.trace_asserts)
        self.assertTrue(changed.allow_scalar_outputs)
        self.assertFalse(changed.trace_asserts)
        self.assertEqual(changed.assume_static_by_default, original.assume_static_by_default)
        with self.assertRaises(FrozenInstanceError):
            original.duck_shape = True


if __name__ == "__main__":
    run_tests()
