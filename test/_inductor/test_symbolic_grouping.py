import torch
from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor.runtime.symbolic_grouping import (
    UnsupportedGroupedPlan,
    build_group_representatives,
    estimate_grouped_benchmark_footprint,
    evaluate_grouped_benchmark_expr,
    required_storage_numel,
)


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


if __name__ == "__main__":
    run_tests()
