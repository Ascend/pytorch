import torch
from types import SimpleNamespace
from unittest.mock import patch

from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

import torch_npu  # noqa: F401
from torch_npu._inductor.codegen import triton as triton_codegen
from torch_npu._inductor.runtime.symbolic_grouping import (
    UnsupportedGroupedPlan,
    build_group_representatives,
    estimate_grouped_benchmark_footprint,
    evaluate_grouped_benchmark_expr,
    required_storage_numel,
)

device = "npu"


class _FixedMemoryDeviceProperties(SimpleNamespace):
    def __getattr__(self, name):
        return getattr(self._base_properties, name)


class TestGroupedBenchmarkFootprint(TestCase):
    def test_required_storage_numel_matches_rand_strided_layout(self):
        self.assertEqual(required_storage_numel((8, 10), (10, 1)), 80)
        self.assertEqual(required_storage_numel((3, 4), (8, 1)), 20)
        self.assertEqual(required_storage_numel((8, 10), (0, 1)), 10)
        self.assertEqual(required_storage_numel((0, 10), (10, 1)), 0)

    def test_invalid_storage_layout_fails_closed(self):
        with self.assertRaisesRegex(
            UnsupportedGroupedPlan,
            "rank mismatch",
        ):
            required_storage_numel((8, 10), (1,))
        with self.assertRaisesRegex(
            UnsupportedGroupedPlan,
            "negative benchmark tensor stride",
        ):
            required_storage_numel((8,), (-1,))

    def test_expression_evaluation_uses_group_axis_environment(self):
        expr = {
            "add": (
                {"mul": ({"axis_name": "x0"}, {"const": 10})},
                {"const": 3},
            )
        }
        self.assertEqual(evaluate_grouped_benchmark_expr(expr, {"x0": 8}), 83)

    def test_footprint_sums_groups_and_one_mutated_clone_peak(self):
        features = ({
            "name": "pointwise",
            "source": "outer_product",
            "axis_names": ("x0",),
            "buckets": (8,),
        },)
        representatives = build_group_representatives(features, ("x0",), ())
        specs = (
            {
                "kind": "tensor",
                "source": "buffer",
                "name": "out_ptr0",
                "dtype": "torch.float16",
                "device": "npu:0",
                "size_exprs": ({"axis_name": "x0"},),
                "stride_exprs": ({"const": 1},),
            },
            {
                "kind": "tensor",
                "source": "buffer",
                "name": "in_ptr0",
                "dtype": "torch.float16",
                "device": "npu:0",
                "size_exprs": ({"axis_name": "x0"}, {"const": 10}),
                "stride_exprs": ({"const": 10}, {"const": 1}),
            },
            {
                "kind": "size",
                "source": "axis_expr",
                "name": "x0_numel",
                "expr": {"axis_name": "x0"},
            },
        )

        footprint = estimate_grouped_benchmark_footprint(
            representatives,
            specs,
            mutated_arg_names=("out_ptr0",),
        )

        self.assertEqual(footprint.group_bytes, ((0, 176), (1, 352)))
        self.assertEqual(footprint.synthetic_bytes, 528)
        self.assertEqual(footprint.mutated_clone_bytes, 32)
        self.assertEqual(footprint.total_bytes, 560)
        self.assertEqual(footprint.largest_group_id, 1)
        self.assertEqual(footprint.dominant_arg.name, "in_ptr0")
        self.assertEqual(footprint.dominant_arg.num_bytes, 320)

    def test_runtime_dependent_expression_fails_closed(self):
        representatives = {
            "reachable_group_ids": (0,),
            "benchmark_axis_values_by_group": ((('x0', 8),),),
        }
        specs = ({
            "kind": "tensor",
            "source": "buffer",
            "name": "in_ptr0",
            "dtype": "torch.float16",
            "device": "npu:0",
            "size_exprs": ({"runtime_arg_index": 1},),
            "stride_exprs": ({"const": 1},),
        },)
        with self.assertRaisesRegex(
            UnsupportedGroupedPlan,
            "runtime_arg_index cannot be bounded",
        ):
            estimate_grouped_benchmark_footprint(representatives, specs)


class TestPointwiseSymbolicGrouping(TestCase):
    @staticmethod
    def _benchmark_guard_meta(width):
        return {
            "kernel_name": "pointwise_footprint_guard_test",
            "group_enabled": True,
            "group_template": "pointwise",
            "group_workload": None,
            "primary_group_axis": "x0",
            "axis_names": ("x0",),
            "axis_static_values": (),
            "group_features": ({
                "name": "pointwise",
                "source": "outer_product",
                "axis_names": ("x0",),
                "buckets": (229376,),
            },),
            "mutated_arg_names": (),
            "ordered_arg_specs": ({
                "kind": "tensor",
                "source": "buffer",
                "name": "in_ptr0",
                "dtype": "torch.float16",
                "device": "npu:0",
                "size_exprs": (
                    {"axis_name": "x0"},
                    {"const": width},
                ),
                "stride_exprs": (
                    {"const": width},
                    {"const": 1},
                ),
            },),
        }

    @staticmethod
    def _run_benchmark_guard(meta, total_memory=108 * 1024**3):
        kernel = object.__new__(triton_codegen.NPUIndexTritonKernel)
        properties = SimpleNamespace(total_memory=total_memory)
        with (
            patch.object(torch.npu, "get_device_properties", return_value=properties),
            patch.object(
                triton_codegen.npu_config,
                "symbolic_group_max_benchmark_memory_ratio",
                0.25,
            ),
        ):
            kernel._disable_grouped_autotune_if_benchmark_too_large(
                meta,
                torch.device("npu", 0),
            )
        return kernel

    @parametrize(
        "width, expected_group_enabled",
        ((69876, False), (1, True)),
    )
    def test_benchmark_footprint_guard_updates_pointwise_group(
        self,
        width,
        expected_group_enabled,
    ):
        meta = self._benchmark_guard_meta(width)
        self._run_benchmark_guard(meta)

        self.assertEqual(meta["group_enabled"], expected_group_enabled)
        if not expected_group_enabled:
            self.assertIsNone(meta["group_template"])

    def test_benchmark_footprint_budget_boundary_is_inclusive(self):
        from torch_npu._inductor.runtime.symbolic_grouping import (
            build_group_representatives,
            estimate_grouped_benchmark_footprint,
        )

        meta = self._benchmark_guard_meta(1)
        representatives = build_group_representatives(
            meta["group_features"],
            meta["axis_names"],
            meta["axis_static_values"],
        )
        footprint = estimate_grouped_benchmark_footprint(
            representatives,
            meta["ordered_arg_specs"],
        )
        self._run_benchmark_guard(meta, footprint.total_bytes * 4)

        self.assertTrue(meta["group_enabled"])

    def test_benchmark_footprint_rejection_enables_auto_blockify(self):
        meta = self._benchmark_guard_meta(69876)
        kernel = self._run_benchmark_guard(meta)
        with (
            patch.object(kernel, "_has_dynamic_shape_axis", return_value=True),
            patch.object(
                triton_codegen.npu_config,
                "enable_symbolic_shape_group_autotune",
                True,
            ),
        ):
            kernel._enable_auto_blockify_for_grouped_fallback_if_needed(meta)

        self.assertFalse(meta["group_enabled"])
        self.assertTrue(meta["enable_auto_blockify"])

    def test_wide_backing_storage_falls_back_before_grouped_benchmark(self):
        import torch_npu._inductor.config as npu_config

        def fn(values):
            return (
                values[:, 6826]
                + values[:, 42400]
                + values[:, 43912]
            )

        values = torch.randn(
            (200, 69876),
            device=device,
            dtype=torch.float16,
        )
        torch._dynamo.mark_dynamic(values, 0)
        expected = fn(values.float()).to(values.dtype)
        fixed_properties = _FixedMemoryDeviceProperties(
            total_memory=108 * 1024**3,
            _base_properties=torch.npu.get_device_properties(torch.device("npu", 0)),
        )

        try:
            with (
                patch.object(
                    torch.npu,
                    "get_device_properties",
                    return_value=fixed_properties,
                ),
                patch.object(
                    npu_config,
                    "enable_symbolic_shape_group_autotune",
                    True,
                ),
                patch.object(
                    npu_config,
                    "symbolic_group_max_benchmark_memory_ratio",
                    0.25,
                ),
            ):
                compiled = torch.compile(fn, backend="inductor", dynamic=True)
                actual, codes = run_and_get_code(compiled, values)
        finally:
            torch._dynamo.reset()

        torch.testing.assert_close(actual, expected)
        matching_codes = [
            code
            for code in codes
            if "69876" in code
            and "'group_enabled': False" in code
            and "'enable_auto_blockify': True" in code
        ]
        self.assertTrue(
            matching_codes,
            f"Expected wide pointwise grouped fallback, got:\n{codes}",
        )


instantiate_parametrized_tests(TestPointwiseSymbolicGrouping)

if __name__ == "__main__":
    run_tests()
