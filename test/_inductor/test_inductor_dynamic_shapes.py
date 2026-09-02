import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sympy
import torch
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

import torch_npu
from torch_npu._inductor.codegen import split_tiling as split_tiling_module
from torch_npu._inductor.codegen.split_tiling import SplitTiling
from torch_npu._inductor.config import num_vector_core
from torch_npu._inductor.runtime import triton_heuristics


if not torch_npu.npu.is_available():
    raise unittest.SkipTest("NPU is not available")

device = "npu"


def make_axis(name, length):
    symbol = sympy.Symbol(name)
    return SimpleNamespace(
        name=name,
        prefix=name[0],
        length=length,
        symbol=lambda: symbol,
    )


class TestSymbolicGroupElementwise(TestCase):
    def setUp(self):
        super().setUp()
        import torch_npu._inductor.config as npu_config

        self.npu_config = npu_config
        self.prev_group_autotune = npu_config.enable_symbolic_shape_group_autotune
        npu_config.enable_symbolic_shape_group_autotune = True
        torch._dynamo.reset()

    def tearDown(self):
        self.npu_config.enable_symbolic_shape_group_autotune = self.prev_group_autotune
        torch._dynamo.reset()
        super().tearDown()

    def _run_and_check(self, fn, inputs, next_inputs, expected_workload):
        for current, next_value in zip(inputs, next_inputs):
            if (
                isinstance(current, torch.Tensor)
                and current.ndim
                and current.shape[0] != next_value.shape[0]
            ):
                torch._dynamo.mark_dynamic(current, 0)

        expected = fn(*inputs)
        compiled = torch.compile(fn, backend="inductor")
        actual, codes = run_and_get_code(compiled, *inputs)
        torch.testing.assert_close(actual, expected)

        next_expected = fn(*next_inputs)
        next_actual = compiled(*next_inputs)
        torch.testing.assert_close(next_actual, next_expected)

        expected_metadata = f"'group_workload': {expected_workload!r}"
        matching_codes = [
            code
            for code in codes
            if "'group_enabled': True" in code
            and "'group_template': 'pointwise'" in code
            and expected_metadata in code
        ]
        self.assertTrue(
            matching_codes,
            f"Expected pointwise group metadata with {expected_metadata}, got:\n{codes}",
        )

    def test_basic_elementwise_workload(self):
        def fn(x, y):
            return torch.relu(x + y) * 0.5

        inputs = (
            torch.randn((257, 1031), device=device),
            torch.randn((257, 1031), device=device),
        )
        next_inputs = (
            torch.randn((263, 1031), device=device),
            torch.randn((263, 1031), device=device),
        )
        self._run_and_check(fn, inputs, next_inputs, "elementwise")

    def test_broadcast_is_not_elementwise_workload(self):
        def fn(x, bias):
            return torch.relu(x + bias) * 0.5

        bias = torch.randn((1031,), device=device)
        inputs = (torch.randn((257, 1031), device=device), bias)
        next_inputs = (torch.randn((263, 1031), device=device), bias)
        self._run_and_check(fn, inputs, next_inputs, None)

    def test_consumed_full_is_elementwise_workload(self):
        def fn(x):
            generated = torch.full_like(x, 2.0)
            return torch.relu(x + generated)

        inputs = (torch.randn((257, 1031), device=device),)
        next_inputs = (torch.randn((263, 1031), device=device),)
        self._run_and_check(fn, inputs, next_inputs, "elementwise")

    def test_standalone_full_is_not_elementwise_workload(self):
        def fn(x):
            return torch.full_like(x, 2.0)

        inputs = (torch.randn((257, 1031), device=device),)
        next_inputs = (torch.randn((263, 1031), device=device),)
        self._run_and_check(fn, inputs, next_inputs, None)

    def test_strided_reindex_is_not_elementwise_workload(self):
        def fn(x, y):
            return torch.relu(x[:, ::2] + y)

        inputs = (
            torch.randn((257, 2062), device=device),
            torch.randn((257, 1031), device=device),
        )
        next_inputs = (
            torch.randn((263, 2062), device=device),
            torch.randn((263, 1031), device=device),
        )
        self._run_and_check(fn, inputs, next_inputs, None)


class TestPointwiseSymbolicGrouping(TestCase):
    def test_static_split_dynamic_tiling_group(self):
        split_axis = make_axis("x0", sympy.Integer(128))
        tiling_axis = make_axis("x1", sympy.Symbol("s0", positive=True))
        kernel = SimpleNamespace(
            persistent_reduction=False,
            inside_reduction=False,
            sorted_axis=[split_axis, tiling_axis],
            split_axis=[split_axis],
            tiling_axis=[tiling_axis],
            features=SimpleNamespace(scheduler_nodes=lambda: ()),
        )
        split_tiling = object.__new__(SplitTiling)
        split_tiling.kernel = kernel
        x0, x1 = (axis.symbol() for axis in kernel.sorted_axis)
        dynamic_stride = sympy.Symbol("s1", positive=True)
        split_tiling.indexing = [x1 + 128 * x0, x0 + dynamic_stride * x1]

        sizevars = SimpleNamespace(size_hint=lambda expr: 64)
        virtualized = SimpleNamespace(graph=SimpleNamespace(sizevars=sizevars))
        with patch.object(split_tiling_module, "V", virtualized):
            self.assertEqual(split_tiling._pointwise_layout_kind(), "transpose")
            meta = split_tiling._build_grouped_meta()

        self.assertIsNotNone(meta)
        self.assertEqual(meta.primary_group_axis, "x1")
        self.assertEqual(meta.static_split_axes, ("x0",))
        self.assertEqual(meta.runtime_block_arg_names, ("X0BLOCK",))
        self.assertEqual(meta.group_features[0].source, "axis")
        self.assertEqual(meta.group_features[0].axis_names, ("x1",))
        self.assertEqual(meta.group_features[0].buckets, (64, 128, 256, 512))

    def test_static_split_dynamic_tiling_broadcast_group(self):
        split_axis = make_axis("x0", sympy.Integer(128))
        tiling_axis = make_axis("x1", sympy.Symbol("s0", positive=True))
        kernel = SimpleNamespace(
            persistent_reduction=False,
            inside_reduction=False,
            sorted_axis=[split_axis, tiling_axis],
            split_axis=[split_axis],
            tiling_axis=[tiling_axis],
            features=SimpleNamespace(scheduler_nodes=lambda: ()),
        )
        split_tiling = object.__new__(SplitTiling)
        split_tiling.kernel = kernel
        x0, x1 = (axis.symbol() for axis in kernel.sorted_axis)
        split_tiling.indexing = [x0, x0 + 128 * x1]

        self.assertEqual(split_tiling._pointwise_layout_kind(), "broadcast")
        meta = split_tiling._build_grouped_meta()

        self.assertIsNotNone(meta)
        self.assertEqual(meta.primary_group_axis, "x1")
        self.assertEqual(meta.group_features[0].name, "pointwise_broadcast_axis")
        self.assertEqual(meta.group_features[0].source, "axis")
        self.assertEqual(meta.group_features[0].axis_names, ("x1",))
        self.assertEqual(meta.group_features[0].buckets, (16, 64, 256, 1024, 4096))

        combined = split_tiling._build_group_features(
            None, tiling_axis, "transpose_broadcast"
        )
        self.assertEqual(combined[0].name, "pointwise_broadcast_axis")
        self.assertEqual(combined[0].buckets, (16, 64, 256, 1024, 4096))

    def test_default_pointwise_group_feature(self):
        primary_axis = make_axis("x0", sympy.Symbol("s0", positive=True))
        kernel = SimpleNamespace(
            persistent_reduction=False,
            inside_reduction=False,
            sorted_axis=[primary_axis],
            split_axis=[primary_axis],
            tiling_axis=[],
            get_axis_dtype=lambda axis: torch.float32,
        )
        split_tiling = object.__new__(SplitTiling)
        split_tiling.kernel = kernel

        features = split_tiling._build_group_features(
            None,
            primary_axis,
            dynamic_split_axes=(primary_axis,),
        )

        vector_core = int(num_vector_core)
        lower = max(1024, split_tiling_module.next_power_of_2(2 * vector_core))
        upper = max(
            split_tiling_module.next_power_of_2(8 * vector_core),
            4096 * vector_core,
        )
        expected_buckets = [lower, upper]
        if upper // lower > 8:
            expected_buckets.append(
                min(
                    split_tiling_module.next_power_of_2(lower * 8),
                    split_tiling_module.next_power_of_2((upper + 1) // 2),
                )
            )
        expected_buckets = tuple(sorted(set(expected_buckets)))

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0].name, "pointwise")
        self.assertEqual(features[0].source, "outer_product")
        self.assertEqual(features[0].axis_names, ("x0",))
        self.assertEqual(features[0].buckets, expected_buckets)

    def test_plain_pointwise_does_not_use_tiling_fallback(self):
        split_axis = make_axis("x0", sympy.Integer(128))
        tiling_axis = make_axis("x1", sympy.Symbol("s0", positive=True))
        kernel = SimpleNamespace(
            persistent_reduction=False,
            inside_reduction=False,
            sorted_axis=[split_axis, tiling_axis],
            split_axis=[split_axis],
            tiling_axis=[tiling_axis],
            features=SimpleNamespace(scheduler_nodes=lambda: ()),
        )
        split_tiling = object.__new__(SplitTiling)
        split_tiling.kernel = kernel
        x0, x1 = (axis.symbol() for axis in kernel.sorted_axis)
        split_tiling.indexing = [x1 + 128 * x0]

        self.assertIsNone(split_tiling._build_grouped_meta())

    def test_tiling_primary_keeps_static_grid_block(self):
        cfg = {"kwargs": {"X0BLOCK": 32, "X1BLOCK_SUB": 64}}
        with patch.object(triton_heuristics, "config_to_dict", return_value=cfg):
            policy = triton_heuristics.build_grouped_launch_policy(
                group_id=0,
                cfg=object(),
                runtime_block_arg_names=("X0BLOCK",),
                group_features=(),
                primary_group_axis="x1",
                primary_feature_index=0,
                axis_env={"x0": 128, "x1": 64},
                npu_num_vector_core=32,
            )

        self.assertEqual(policy["static_blocks"], (("X0BLOCK", 32),))
        self.assertEqual(policy["runtime_block_rules"], ())
        self.assertEqual(policy["grid_target"], 1)

    def test_pointwise_tiling_fallback_requires_one_dynamic_axis(self):
        x0 = make_axis("x0", sympy.Symbol("s0", positive=True))
        x1 = make_axis("x1", sympy.Symbol("s1", positive=True))
        kernel = SimpleNamespace(
            persistent_reduction=False,
            inside_reduction=False,
            sorted_axis=[x0, x1],
            split_axis=[],
            tiling_axis=[x0, x1],
        )
        split_tiling = object.__new__(SplitTiling)
        split_tiling.kernel = kernel

        self.assertIsNone(split_tiling._dynamic_pointwise_tiling_axis())


instantiate_parametrized_tests(TestSymbolicGroupElementwise)
instantiate_parametrized_tests(TestPointwiseSymbolicGrouping)


if __name__ == "__main__":
    run_tests()
