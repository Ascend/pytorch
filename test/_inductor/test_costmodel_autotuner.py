import sys
import types
from unittest.mock import call, MagicMock, patch

from torch.testing._internal.common_utils import run_tests, TestCase

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.runtime import triton_heuristics


class TestCostModelAutotuner(TestCase):
    def test_costmodel_ranks_and_partitions_configs(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        configs = tuple(object() for _ in range(4))
        autotuner.configs = list(configs)
        autotuner.heuristic_type = triton_heuristics.HeuristicType.POINTWISE
        autotuner._costmodel_fallback_configs = None
        costmodel_items = tuple(object() for _ in configs)
        autotuner._build_costmodel_items = MagicMock(
            return_value=list(costmodel_items)
        )

        costmodel_runtime = types.ModuleType(
            "triton.backends.ascend.runtime.costmodel_runtime"
        )
        costmodel_runtime.costmodel_bench = MagicMock(
            side_effect=[
                (configs[0], 3.0),
                (configs[1], float("inf")),
                (configs[2], 1.0),
                (configs[3], 2.0),
            ]
        )

        with (
            patch.dict(
                sys.modules,
                {costmodel_runtime.__name__: costmodel_runtime},
            ),
            patch.object(npu_config, "enable_costmodel_prefilter", True),
            patch.object(npu_config, "costmodel_ratio", 0.5),
            patch.object(npu_config, "precompile_thread_num", 1),
        ):
            autotuner._apply_costmodel_to_configs("runtime-arg")

        self.assertEqual(autotuner.configs, [configs[2], configs[3]])
        self.assertEqual(autotuner._costmodel_fallback_configs, [configs[0]])
        autotuner._build_costmodel_items.assert_called_once_with(
            ("runtime-arg",), {}
        )
        costmodel_runtime.costmodel_bench.assert_has_calls(
            [call(item) for item in costmodel_items]
        )

    def test_costmodel_retries_filtered_configs_after_compile_failure(self):
        autotuner = object.__new__(triton_heuristics.NPUCostModelAutotuner)
        autotuner.configs = ["selected"]
        autotuner.candidate_plan = {"plan": "selected"}
        autotuner.runtime_block_arg_names = ()
        autotuner._costmodel_fallback_configs = ["fallback"]
        autotuner._precompile_variant_configs = MagicMock(
            side_effect=lambda configs: [f"compile-{configs[0]}"]
        )
        autotuner.get_fn_name = MagicMock(return_value="test_kernel")

        compile_fn = MagicMock(
            side_effect=[
                triton_heuristics.NoTritonConfigsError("selected failed"),
                ["fallback result"],
            ]
        )
        fallback_plan = {"plan": "fallback"}

        with patch.object(
            triton_heuristics,
            "build_candidate_plan",
            return_value=fallback_plan,
        ) as build_plan:
            autotuner._precompile_current_configs(compile_fn)

        compile_fn.assert_has_calls(
            [call(["compile-selected"]), call(["compile-fallback"])],
        )
        self.assertEqual(compile_fn.call_count, 2)
        build_plan.assert_called_once_with(["fallback"], ())
        self.assertEqual(autotuner.compile_results, ["fallback result"])
        self.assertEqual(autotuner.configs, ["fallback"])
        self.assertIs(autotuner.candidate_plan, fallback_plan)
        self.assertIsNone(autotuner._costmodel_fallback_configs)


if __name__ == "__main__":
    run_tests()
