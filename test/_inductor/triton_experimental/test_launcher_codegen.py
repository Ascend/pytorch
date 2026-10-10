# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""CPU-only tests of the actual dependency-free launcher source generator.

Load the leaf module directly to avoid importing the NPU runtime package.
No production function is extracted or replaced; these tests do not test CANN.
"""

import importlib.util
import inspect
import unittest
from pathlib import Path
from types import FunctionType, SimpleNamespace


_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "torch_npu/_inductor/triton_experimental/launcher_codegen.py"
)
_SPEC = importlib.util.spec_from_file_location("npu_launcher_codegen_under_test", _SOURCE)
codegen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(codegen)


class TestLauncherCodegen(unittest.TestCase):
    def make_launcher(self, names, meta, *, arg_names=None, blocks=None, is_a5=False,
                      bound_names=(), runner=None, num_cores=40):
        calls = []
        scope = {"runner": calls.append if runner is None else runner}
        grid = codegen._gen_grid_code(
            scope, names, names if arg_names is None else arg_names,
            SimpleNamespace(kwargs={"XBLOCK": 128} if blocks is None else blocks),
            meta, num_cores=num_cores, is_a5=is_a5,
        )
        # Pass one tuple to the recording runner; the generator does not choose ABI.
        arguments = ["(grid_0, grid_1, grid_2, stream)"]
        launcher = codegen._gen_launcher_code(
            scope, names, arguments, grid, bound_names=bound_names,
        )
        return launcher, calls

    def test_dynamic_grid_clamp_and_cache(self):
        for node_count in (0, 1):
            for bindings in ((), ("runner", "max", "min", "_grid_cache")):
                with self.subTest(nodes=node_count, bindings=bindings):
                    launcher, calls = self.make_launcher(
                        ["xnumel"], {"npu_num_x_nodes": node_count},
                        bound_names=bindings,
                    )
                    for count in (0, 1, 129, 129, 10000, 257):
                        launcher(count, stream=17)
                    self.assertEqual([call[0] for call in calls], [1, 1, 2, 2, 40, 3])
                    self.assertEqual(launcher.__globals__["_grid_cache"], [257, 3])
                    self.assertEqual(launcher._expected_positional_count, 1)
                    self.assertEqual(launcher._npu_def_args, ["xnumel"])

    def test_grid_fallback_and_reduction(self):
        cases = [
            ({}, ["xnumel"], 40),
            ({"npu_num_x_nodes": 2}, ["xnumel"], 40),
            ({"npu_num_x_nodes": 1, "grid_type": "Grid2D"}, ["xnumel"], 40),
            ({"npu_num_x_nodes": 1, "grid_type": "Grid3D"}, ["xnumel"], 40),
            ({"npu_num_x_nodes": 0}, ["xnumel", "R0_BLOCK"], 1),
            ({"npu_num_x_nodes": 0, "npu_rsplit_partial": True}, ["xnumel", "R0_BLOCK"], 40),
            ({"npu_num_x_nodes": 0}, ["xnumel", "R0_BLOCK", "ws_ptr"], 40),
        ]
        for meta, arg_names, expected in cases:
            with self.subTest(meta=meta, arg_names=arg_names):
                launcher, calls = self.make_launcher(["xnumel"], meta, arg_names=arg_names)
                launcher(1, stream=17)
                self.assertEqual(calls, [(expected, 1, 1, 17)])
                self.assertNotIn("_grid_cache", launcher.__globals__)

    def test_te_combo_uses_full_core_grid(self):
        meta = {
            "te_combo_meta": {"combo_num_kernels": 2},
            "npu_num_x_nodes": 1,
            "npu_dispatch_recipe": {
                "lines": ["blocks = (xnumel + XBLOCK - 1) // XBLOCK"],
                "factors": ["blocks"],
            },
        }
        launcher, calls = self.make_launcher(
            ["xnumel", "ynumel"], meta,
            blocks={"XBLOCK": 128}, is_a5=True, num_cores=48,
        )
        launcher(1, 10000, stream=17)
        launcher(10000, 1, stream=19)
        self.assertEqual([call[0] for call in calls], [48, 48])
        self.assertNotIn("_grid_cache", launcher.__globals__)

    def test_non_linearize_reduction_uses_exact_grid(self):
        # The same shape as upstream's non-linearize sum regression: 64*128
        # output rows. A physical-core cap misses rows; overlaunch can overread.
        for node_count in (None, 0, 1):
            for block, expected in ((128, 64), (512, 16)):
                for bindings in ((), ("runner", "max", "_grid_cache")):
                    with self.subTest(nodes=node_count, block=block, bindings=bindings):
                        meta = {"npu_linearize": False, "grid_type": "Grid1D"}
                        if node_count is not None:
                            meta["npu_num_x_nodes"] = node_count
                        launcher, calls = self.make_launcher(
                            ["xnumel"], meta, arg_names=["out", "xnumel", "R0_BLOCK"],
                            blocks={"XBLOCK": block}, num_cores=48,
                            bound_names=bindings,
                        )
                        launcher(8192, stream=17)
                        self.assertEqual(calls, [(expected, 1, 1, 17)])

    def test_non_linearize_grid_cache_and_runner_clone(self):
        launcher, calls = self.make_launcher(
            ["xnumel"], {"npu_linearize": False, "npu_num_x_nodes": 0},
            arg_names=["out", "xnumel", "R0_BLOCK"], num_cores=48,
        )
        counts = (0, 1, 129, 8192, 8192, 257)
        for count in counts:
            launcher(count, stream=17)
        self.assertEqual([call[0] for call in calls], [1, 1, 2, 64, 64, 3])
        self.assertEqual(launcher.__globals__["_grid_cache"], [257, 3])
        fast_calls = []
        clone = FunctionType(
            launcher.__code__, {**launcher.__globals__, "runner": fast_calls.append},
        )
        clone(8192, stream=19)
        self.assertEqual(fast_calls, [(64, 1, 1, 19)])
        self.assertEqual(len(calls), len(counts))

    def test_exact_grid_does_not_change_other_dispatch_modes(self):
        cases = [
            # Missing flag keeps the legacy linearize behavior.
            ({"npu_num_x_nodes": 0}, ["xnumel"], ["xnumel", "R0_BLOCK"], 1),
            ({"npu_linearize": True, "npu_num_x_nodes": 0},
             ["xnumel"], ["xnumel", "R0_BLOCK"], 1),
            ({"npu_linearize": True, "npu_num_x_nodes": 1},
             ["xnumel"], ["xnumel", "R0_BLOCK"], 48),
            ({"npu_linearize": False, "grid_type": "Grid2D"},
             ["xnumel"], ["xnumel", "R0_BLOCK"], 48),
            ({"npu_linearize": False, "grid_type": "Grid3D"},
             ["xnumel"], ["xnumel", "R0_BLOCK"], 48),
            ({"npu_linearize": False}, [], ["R0_BLOCK"], 48),
        ]
        for meta, names, arg_names, expected in cases:
            with self.subTest(meta=meta, names=names):
                launcher, calls = self.make_launcher(
                    names, meta, arg_names=arg_names, num_cores=48,
                )
                launcher(*([8192] if names else []), stream=17)
                self.assertEqual(calls, [(expected, 1, 1, 17)])

    def test_a5_recipe_uses_all_dimensions_without_single_slot_cache(self):
        meta = {
            "grid_type": "Grid3D",
            "npu_dispatch_recipe": {
                "lines": [
                    "xb = (xnumel + XBLOCK - 1) // XBLOCK",
                    "yb = (ynumel + YBLOCK - 1) // YBLOCK",
                    "zb = (znumel + ZBLOCK - 1) // ZBLOCK",
                ],
                "factors": ["xb", "yb", "zb"],
            },
        }
        launcher, calls = self.make_launcher(
            ["xnumel", "ynumel", "znumel"], meta,
            blocks={"XBLOCK": 128, "YBLOCK": 16, "ZBLOCK": 4}, is_a5=True,
        )
        for sizes in ((129, 17, 5), (129, 33, 5), (0, 33, 5)):
            launcher(*sizes, stream=17)
        self.assertEqual([call[0] for call in calls], [8, 12, 1])
        self.assertNotIn("_grid_cache", launcher.__globals__)
        fallback, fallback_calls = self.make_launcher(
            ["xnumel", "ynumel", "znumel"], meta, is_a5=False,
        )
        fallback(1, 1, 1, stream=17)
        self.assertEqual(fallback_calls[0][0], 40)

    def test_global_runner_clone_does_not_rebind_baseline(self):
        launcher, baseline_calls = self.make_launcher(["xnumel"], {"npu_num_x_nodes": 1})
        fast_calls = []
        clone = FunctionType(
            launcher.__code__, {**launcher.__globals__, "runner": fast_calls.append},
        )
        self.assertEqual(list(inspect.signature(launcher).parameters), ["xnumel", "stream"])
        self.assertIsNone(launcher.__defaults__)
        self.assertIsNone(launcher.__closure__)
        launcher(129, stream=17)
        clone(257, stream=19)
        self.assertEqual(baseline_calls, [(2, 1, 1, 17)])
        self.assertEqual(fast_calls, [(3, 1, 1, 19)])

    def test_bound_runner_is_independent_of_globals(self):
        launcher, calls = self.make_launcher([], {}, bound_names=("runner",))
        launcher.__globals__["runner"] = lambda *args: self.fail("bound runner was replaced")
        launcher(stream=17)
        self.assertEqual(calls, [(40, 1, 1, 17)])
        self.assertIn("runner", inspect.signature(launcher).parameters)

    def test_caller_supplied_fallback_statements(self):
        fast_calls, slow_calls = [], []

        def runner(*args):
            fast_calls.append(args)
            return -1 if args[-1] else None

        scope = {"runner": runner, "slow_runner": lambda *args: slow_calls.append(args)}
        runner_args = ["grid_0", "stream", "decline"]
        launcher = codegen._gen_launcher_code(
            scope, ["decline"], runner_args, ["    grid_0 = 1"],
            bound_names=("runner", "slow_runner"),
            call_lines=[
                "    if runner(grid_0, stream, decline) is not None:",
                "        slow_runner(grid_0, stream, decline)",
            ],
        )
        launcher(False, stream=17)
        launcher(True, stream=17)
        self.assertEqual(fast_calls, [(1, 17, False), (1, 17, True)])
        self.assertEqual(slow_calls, [(1, 17, True)])

    def test_no_runtime_arguments(self):
        launcher, calls = self.make_launcher([], {})
        launcher(stream=17)
        self.assertEqual(list(inspect.signature(launcher).parameters), ["stream"])
        self.assertEqual(launcher._expected_positional_count, 0)
        self.assertEqual(calls, [(40, 1, 1, 17)])


if __name__ == "__main__":
    unittest.main()
