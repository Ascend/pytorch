import contextlib
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import sympy
import torch
from torch._inductor import config
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import run_tests

from testutils import TestUtils

import torch_npu  # noqa: F401
from torch_npu._inductor.config import catlass as catlass_config
from torch_npu._inductor.select_algorithm import (
    NPUTemplateCompileOption,
    NPUTritonBenchmarkRequest,
    NPUTritonRuntimeRenderer,
    NPUTritonTemplate,
    _add_npu_template_compile_options_to_triton_meta,
    _add_npu_template_meta_to_inductor_meta,
    _extract_choice_debug_config,
    _format_choice_debug_label,
    _gen_npu_template_triton_imports,
    _log_autotune_error,
    _select_first_usable_choice_in_order,
    _summarize_autotune_error,
    _tiling_sort_key,
)


class _Choice:
    """Choice stand-in exposing the attributes the selection helpers read."""

    def __init__(self, name, config_info=None, hash_key=None):
        self.name = name
        self.config = config_info
        self._hash_key = hash_key if hash_key is not None else name

    def hash_key(self):
        return self._hash_key


class TestTemplateMetaHelpers(TestUtils):
    def test_template_meta_helpers(self):
        option = NPUTemplateCompileOption({"ENABLE_COMPILE_HINT": True, "BLOCK_M": 64})
        meta = {"BLOCK_M": 32}
        option.apply(meta)
        self.assertEqual(meta, {"BLOCK_M": 32, "ENABLE_COMPILE_HINT": True})

        inductor_meta = {}
        meta = {"ENABLE_COMPILE_HINT": True, "BLOCK_M": 16, "BLOCK_N": 32, "other": 1}
        _add_npu_template_meta_to_inductor_meta(inductor_meta, meta)
        self.assertEqual(inductor_meta, {"ENABLE_COMPILE_HINT": True, "BLOCK_M": 16, "BLOCK_N": 32})

        # Compile options are filtered against the allowed keys and then merged.
        triton_meta = {"npu_compile_options": {"EXISTING": 1}}
        _add_npu_template_compile_options_to_triton_meta(triton_meta, {"A": 1, "B": 2, "C": 3},
                                                         frozenset({"A", "C"}))
        self.assertEqual(triton_meta["npu_compile_options"], {"EXISTING": 1, "A": 1, "C": 3})
        triton_meta = {}
        _add_npu_template_compile_options_to_triton_meta(triton_meta, {"A": 1}, frozenset({"B"}))
        self.assertNotIn("npu_compile_options", triton_meta)

        first = _gen_npu_template_triton_imports()
        self.assertIn("torch_npu._inductor.runtime", first)
        self.assertIn("triton_heuristics", first)
        self.assertEqual(first, _gen_npu_template_triton_imports())


class TestChoiceDebugHelpers(TestUtils):
    def test_choice_debug_helpers(self):
        choice = SimpleNamespace(_sparse_mask_report_config=None, log_info="autotune-log",
                                 config={"BLOCK_M": 8})
        self.assertEqual(_extract_choice_debug_config(choice), "autotune-log")
        choice.log_info = None
        self.assertEqual(_extract_choice_debug_config(choice), {"BLOCK_M": 8})
        self.assertIsNone(_extract_choice_debug_config(SimpleNamespace(config=None)))

        choice = SimpleNamespace(name="triton_mm", config={"BLOCK_M": 32})
        self.assertEqual(_format_choice_debug_label(choice), "triton_mm config={'BLOCK_M': 32}")
        # An object without any debug config must fall back to str(choice).
        plain = object()
        label = _format_choice_debug_label(plain)
        self.assertNotIn("config=", label)
        self.assertEqual(label, str(plain))

        error = "\n".join("line%d" % index for index in range(1, 13))
        summary = _summarize_autotune_error(error, max_lines=8)
        self.assertTrue(summary.startswith("... <earlier compiler diagnostic suppressed>"))
        self.assertIn("line12", summary)
        self.assertNotIn("line3", summary)
        self.assertEqual(_summarize_autotune_error("only line"), "only line")
        self.assertEqual(_summarize_autotune_error(ValueError("")), "ValueError")
        self.assertEqual(_summarize_autotune_error("  \n\n"), "empty error")


class TestTilingSelection(TestUtils):
    def test_sort_key(self):
        self.assertEqual(_tiling_sort_key(_Choice("single", {"BLOCK_M": 32, "BLOCK_N": 64})),
                         (2048, 32, 64))
        self.assertEqual(_tiling_sort_key(_Choice("masked", {"MASK_BLOCK_M": 16, "MASK_BLOCK_N": 32,
                                                             "BLOCK_M": 64})), (512, 16, 32))
        self.assertEqual(_tiling_sort_key(_Choice("partial", {"BLOCK_M": "32"})), (32, 32, 1))
        self.assertEqual(_tiling_sort_key(_Choice("dual", {"BLOCK_M": 32, "BLOCK_N": 64,
                                                           "BLOCK_M2": 16, "BLOCK_N2": 32})),
                         (2048 + 512, 2048, 512, 32, 64, 16, 32))
        self.assertIsNone(_tiling_sort_key(_Choice("empty", None)))
        self.assertIsNone(_tiling_sort_key(_Choice("second", {"BLOCK_M2": 16})))
        self.assertIsNone(_tiling_sort_key(_Choice("scalar", 7)))

    def test_selection(self):
        measured, other = _Choice("measured"), _Choice("other")
        self.assertIs(_select_first_usable_choice_in_order([measured, other], {other: 1.0}, set()),
                      other)
        first, second = _Choice("first"), _Choice("second", hash_key="precompiled")
        self.assertIs(_select_first_usable_choice_in_order([first, second], {}, {"precompiled"}),
                      second)
        small = _Choice("small", {"BLOCK_M": 32, "BLOCK_N": 64})
        large = _Choice("large", {"BLOCK_M": 64, "BLOCK_N": 64})
        self.assertIs(_select_first_usable_choice_in_order([small, large], {}, set()), large)
        self.assertIsNone(_select_first_usable_choice_in_order([_Choice("a", None),
                                                               _Choice("b", None)], {}, set()))


class TestAutotuneLogging(TestUtils):
    def test_error_log_marks_ignored_choice(self):
        choice = SimpleNamespace(name="triton_mm", config={"BLOCK_M": 32})
        with self.assertLogs("torch._inductor", level="ERROR") as logs:
            _log_autotune_error("benchmark failed", "first\nsecond", choice, ignored=True)
        self.assertEqual(len(logs.records), 1)
        message = logs.output[0]
        self.assertIn("triton_mm config={'BLOCK_M': 32}", message)
        self.assertIn("ignoring this choice", message)
        self.assertIn("second", message)

    def test_full_diagnostic_only_logged_at_debug(self):
        choice = SimpleNamespace(name="triton_mm", config={"BLOCK_M": 32})
        with self.assertLogs("torch._inductor", level="DEBUG") as logs:
            _log_autotune_error("benchmark failed", RuntimeError("bad kernel"), choice, ignored=False)
        self.assertEqual(len(logs.records), 2)
        self.assertNotIn("ignoring this choice", logs.output[0])


class TestRuntimeRenderer(TestUtils):
    def setUp(self):
        graph = SimpleNamespace(
            get_dtype=lambda name: "graph-dtype:%s" % name,
            get_buffer=lambda name: "graph-buffer:%s" % name,
            scheduler=SimpleNamespace(get_buffer_layout=lambda name: "graph-layout:%s" % name),
        )
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(V.set_graph_handler(graph))
        self.runtime_buffer = SimpleNamespace(get_layout=lambda: "runtime-layout")
        self.renderer = NPUTritonRuntimeRenderer(
            kernel=SimpleNamespace(args=SimpleNamespace(python_argdefs=lambda: ("args", "defs"))),
            render=lambda: None,
            runtime_arg_names={"arg1": "runtime1"},
            runtime_arg_dtypes={"arg1": torch.float16},
            runtime_arg_buffers={"arg1": self.runtime_buffer},
        )

    def test_runtime_args_override_graph_lookups(self):
        with self.renderer.patch_runtime_args():
            self.assertEqual(V.graph.get_dtype("arg1"), torch.float16)
            self.assertEqual(V.graph.get_dtype("other"), "graph-dtype:other")
            self.assertIs(V.graph.get_buffer("arg1"), self.runtime_buffer)
            self.assertEqual(V.graph.scheduler.get_buffer_layout("arg1"), "runtime-layout")
            self.assertEqual(V.graph.scheduler.get_buffer_layout("other"), "graph-layout:other")
        # The graph lookups are restored when the context exits.
        self.assertEqual(V.graph.get_dtype("arg1"), "graph-dtype:arg1")
        self.assertEqual(V.graph.scheduler.get_buffer_layout("arg1"), "graph-layout:arg1")

    def test_python_argdefs_runs_under_patched_args(self):
        self.assertEqual(self.renderer.python_argdefs(), ("args", "defs"))


class TestBenchmarkRequestValidation(TestUtils):
    def test_bad_output_tensors_are_rejected(self):
        request = object.__new__(NPUTritonBenchmarkRequest)
        with self.assertRaises(ValueError):
            request.make_run_fn(out=torch.zeros(4), output_tensor=torch.zeros(4))
        with self.assertRaises(ValueError):
            request.make_run_fn()


class TestIndexDtypeDefine(TestUtils):
    def test_index_dtype_defines(self):
        template = object.__new__(NPUTritonTemplate)
        template.name = "mm_template"
        defines = StringIO()
        with mock.patch.object(TritonScheduling, "can_use_32bit_indexing", return_value=True):
            template._write_index_dtype_define(defines, sympy.Integer(64), {})
        self.assertEqual(defines.getvalue(), "INDEX_DTYPE : tl.constexpr = tl.int32\n")

        template = object.__new__(NPUTritonTemplate)
        template.name = "flex_attention_fwd"
        defines = StringIO()
        with mock.patch.object(TritonScheduling, "can_use_32bit_indexing", return_value=False):
            template._write_index_dtype_define(defines, sympy.Integer(1 << 33), {})
        self.assertEqual(defines.getvalue(), "INDEX_DTYPE : tl.constexpr = tl.int64\n")

        # A symbolic extent cannot be proven to fit in 32 bits, so it falls back
        # to 64-bit indexing instead of failing the compile.
        template = object.__new__(NPUTritonTemplate)
        template.name = "mm_template"
        defines = StringIO()
        with mock.patch.object(TritonScheduling, "can_use_32bit_indexing", return_value=False):
            index_dtype = template._write_index_dtype_define(defines, sympy.Symbol("s0"), {})
        self.assertEqual(index_dtype, "tl.int64")
        self.assertEqual(defines.getvalue(), "INDEX_DTYPE : tl.constexpr = tl.int64\n")


class TestSelectAlgorithmCompile(TestUtils):
    @config.patch(max_autotune=True, force_disable_caches=True)
    def test_max_autotune_mm_compiles_and_matches(self):
        def fn(mat1, mat2):
            return torch.mm(mat1, mat2)

        mat1 = torch.randn(128, 256, dtype=torch.float16, device="npu")
        mat2 = torch.randn(256, 128, dtype=torch.float16, device="npu")
        expected = fn(mat1, mat2)
        # Numerics alone cannot tell autotuning apart from a silent fallback:
        # observe the selection path and the candidates it was handed.
        benchmark_calls = []
        orig_make_benchmark_fn = AlgorithmSelectorCache.make_benchmark_fn

        def counting_make_benchmark_fn(cls, choices, *args, **kwargs):
            benchmark_calls.append(choices)
            return orig_make_benchmark_fn(choices, *args, **kwargs)

        with mock.patch.object(AlgorithmSelectorCache, "make_benchmark_fn",
                               classmethod(counting_make_benchmark_fn)):
            actual, _ = run_and_get_code(torch.compile(fn), mat1, mat2)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-2, rtol=1e-2))
        # A silent eager fallback would match numerically but never offer choices.
        self.assertTrue(benchmark_calls, "autotune never built a benchmark")
        self.assertTrue(benchmark_calls[0], "autotune was offered no candidates")


class TestBenchmarkStrategySelection(TestUtils):
    """make_benchmark_fn() dispatches to mutually exclusive strategies."""

    def _strategy(self):
        return AlgorithmSelectorCache.make_benchmark_fn([], [], None)

    def test_default_and_profiling_strategies(self):
        self.assertEqual(self._strategy().__name__, "benchmark_in_current_process")
        with mock.patch.object(catlass_config, "catlass_bench_use_profiling", True):
            strategy = self._strategy()
        self.assertEqual(strategy.__name__, "profiling_choices_in_current_process")

    @config.patch(autotune_in_subproc=True)
    def test_autotune_in_subproc_selects_the_subprocess_strategy(self):
        self.assertEqual(self._strategy().__name__, "benchmark_in_sub_process")


if __name__ == "__main__":
    run_tests()
