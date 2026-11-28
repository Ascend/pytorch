# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext 接口功能正确性
API 名称：torch.fx.experimental.symbolic_shapes.SubclassSymbolicContext
API 签名：SubclassSymbolicContext(dynamic_sizes, dynamic_strides=None, constraint_sizes=None, constraint_strides=None,
    specialize_on=None, view_base_context=None, shape_ids=None, unbacked_bounds=None, inner_contexts=<factory>,
    track_outer_size_stride=True, *, tensor_source, shape_env_to_source_to_symbol_cache=<factory>,
    excluded_sizes=None)
    <factory> 表示默认工厂：inner_contexts=dict；shape_env_to_source_to_symbol_cache=dict，每次调用创建新对象。

覆盖维度表：
| 覆盖维度 | 覆盖情况 |
| --- | --- |
| 空/非空 | 空维度，可选字段 None/空/非空，默认字典独立 |
| 枚举选项 | 五种 size 策略、三种合法 stride 策略和 track_outer_size_stride 两值 |
| 参数类型 | 动态策略、约束/回调、内层/基础上下文、shape_ids/bounds/cache 字典 |
| 传参与不传参 | 可选字段默认与显式值，None/空集合的默认填充 |
| 等价类/边界值 | 空维度、上下文/缓存身份及非法 stride |
| 正常传参场景 | 内层上下文和外层策略默认值/显式值 |
| 异常传参场景 | STATIC stride 策略被明确拒绝 |
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

from torch.fx.experimental.symbolic_shapes import SubclassSymbolicContext
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, SymbolicContext, SymIntSymbolicContext


@unittest.skipUnless(RUN_FLAG, SKIP_REASON)
class TestApi(TestCase):
    def setUp(self):
        super().setUp()
        self.device_name = DEVICE_NAME
        self.device = torch.device(self.device_name) if self.device_name else None

    def test_subclasssymboliccontext_default_outer_policies_and_independent_inner_contexts(self):
        first = SubclassSymbolicContext([DimDynamic.DYNAMIC], tensor_source=LocalSource("first"))
        second = SubclassSymbolicContext([], tensor_source=LocalSource("second"))
        self.assertEqual(first.dynamic_strides, [DimDynamic.INFER_STRIDE])
        self.assertEqual(first.constraint_sizes, [None])
        self.assertTrue(first.track_outer_size_stride)
        first.inner_contexts["inner"] = SymbolicContext()
        self.assertEqual(second.inner_contexts, {})
        self.assertEqual(second.dynamic_strides, [])

    def test_subclasssymboliccontext_explicit_inner_policy_and_invalid_stride_policy(self):
        inner = SymIntSymbolicContext(None)
        context = SubclassSymbolicContext(
            [DimDynamic.STATIC], dynamic_strides=[DimDynamic.DYNAMIC],
            tensor_source=LocalSource("tensor"), inner_contexts={"inner": inner},
            track_outer_size_stride=False)
        self.assertIs(context.inner_contexts["inner"], inner)
        self.assertFalse(context.track_outer_size_stride)
        with self.assertRaisesRegex(AssertionError, "dynamic_strides"):
            SubclassSymbolicContext([DimDynamic.STATIC], dynamic_strides=[DimDynamic.STATIC],
                                    tensor_source=LocalSource("invalid"))


    def test_parameter_explicit_optional_fields_and_all_stride_policies(self):
        import sympy
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.value_ranges import ValueRanges

        constraint = StrictMinMaxConstraint(warn_only=False, vr=ValueRanges(2, 8))
        base = SymbolicContext()
        inner = SymIntSymbolicContext(None)

        def callback(value):
            return value

        for size_policy in DimDynamic:
            for stride_policy in (DimDynamic.INFER_STRIDE, DimDynamic.DYNAMIC, DimDynamic.DUCK):
                with self.subTest(size_policy=size_policy, stride_policy=stride_policy):
                    cache = {7: {"source": sympy.Symbol("cached")}}
                    context = SubclassSymbolicContext(
                        dynamic_sizes=[size_policy], dynamic_strides=[stride_policy],
                        constraint_sizes=[constraint], constraint_strides=[None], specialize_on=[[callback]],
                        view_base_context=base, shape_ids={0: "outer"}, unbacked_bounds={0: (2, None)},
                        tensor_source=LocalSource("tensor"), shape_env_to_source_to_symbol_cache=cache,
                        excluded_sizes=(2, None), inner_contexts={"inner": inner}, track_outer_size_stride=False)
                    self.assertEqual(context.dynamic_sizes, [size_policy])
                    self.assertEqual(context.dynamic_strides, [stride_policy])
                    self.assertIs(context.constraint_sizes[0], constraint)
                    self.assertEqual(context.constraint_strides, [None])
                    self.assertIs(context.specialize_on[0][0], callback)
                    self.assertIs(context.view_base_context, base)
                    self.assertEqual(context.shape_ids, {0: "outer"})
                    self.assertEqual(context.unbacked_bounds, {0: (2, None)})
                    self.assertIs(context.shape_env_to_source_to_symbol_cache, cache)
                    self.assertEqual(context.excluded_sizes, (2, None))
                    self.assertIs(context.inner_contexts["inner"], inner)
                    self.assertFalse(context.track_outer_size_stride)

    def test_parameter_none_empty_defaults_and_invalid_stride_policies(self):
        first = SubclassSymbolicContext(
            [], dynamic_strides=None, constraint_sizes=None, constraint_strides=None, specialize_on=None,
            view_base_context=None, shape_ids=None, unbacked_bounds=None, tensor_source=LocalSource("first"),
            excluded_sizes=None, track_outer_size_stride=True)
        second = SubclassSymbolicContext([], tensor_source=LocalSource("second"),
                                         inner_contexts={}, shape_env_to_source_to_symbol_cache={})
        self.assertEqual(first.dynamic_sizes, [])
        self.assertEqual(first.dynamic_strides, [])
        self.assertEqual(first.constraint_sizes, [])
        self.assertEqual(first.constraint_strides, [])
        self.assertEqual(first.specialize_on, [])
        self.assertIsNone(first.view_base_context)
        self.assertIsNone(first.shape_ids)
        self.assertIsNone(first.unbacked_bounds)
        self.assertIsNone(first.excluded_sizes)
        self.assertTrue(first.track_outer_size_stride)
        first.shape_env_to_source_to_symbol_cache[3] = {}
        self.assertEqual(second.shape_env_to_source_to_symbol_cache, {})
        first.inner_contexts["inner"] = SymbolicContext()
        self.assertEqual(second.inner_contexts, {})
        for policy in (DimDynamic.STATIC, DimDynamic.UNBACKED, None, "invalid"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(AssertionError, "dynamic_strides"):
                    SubclassSymbolicContext([DimDynamic.DYNAMIC], dynamic_strides=[policy],
                                            tensor_source=LocalSource("invalid"))


if __name__ == "__main__":
    run_tests()
