# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Owner(s): ["module: inductor"]
"""Fast launcher contract and device tests.

Reference: local PyTorch b62a09d1dd4, TestFastCudaLauncher and
TestFastCudaLauncherCompileResult. Missing fast support FAILS positive tests.
Only lack of NPU hardware skips device tests; Python tests mock the C++ boundary.
Run queue variants in fresh processes with TASK_QUEUE_ENABLE=0 and =1.
"""

import gc
import weakref
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch
import torch_npu
import triton
import triton.language as tl
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor import config
from torch._inductor.runtime.hints import HeuristicType
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from fast_launcher_test_utils import (
    isolated_fast_config,
    observe_fast_builds,
    require_fast_type,
)
from torch_npu._inductor.triton_experimental import npu_triton_heuristics as heuristics
from torch_npu._inductor.triton_experimental.static_launcher.adapter import (
    NPUStaticArtifactAdapter,
)


def make_autotuner(launcher, *, enabled=True):
    def kernel_fn():
        pass

    kernel_fn.src = "def fast_test_kernel(): pass"
    device = SimpleNamespace(
        type="npu", index=0, warp_size=32, max_threads_per_block=1024,
    )
    tuner = heuristics.NPUCachingAutotuner(
        fn=kernel_fn,
        triton_meta={"device": device, "signature": {}, "constants": {}},
        configs=[triton.Config({}, num_warps=4)],
        save_cache_hook=None, mutated_arg_names=[], optimize_mem=False,
        heuristic_type=HeuristicType.POINTWISE,
        inductor_meta={"use_fast_triton_launcher": enabled},
    )
    tuner.launchers = [launcher]
    return tuner


class FakeKernel:
    """Only for Python binding tests; no claim about C++ or device execution."""

    def __init__(self):
        self.loaded_kernel = object()
        self.function = self.loaded_kernel
        self.num_warps = 4
        self.shared = 0
        self.arg_tys = "OOi"
        self.calls = []

    def run(self, *args):
        self.calls.append(args)


def make_launcher(kernel):
    scope = {"runner": kernel.run}
    exec(
        "def launcher(in_ptr0, out_ptr0, xnumel, stream):\n"
        "    runner(1, 1, 1, stream, in_ptr0, out_ptr0, xnumel)\n",
        scope,
    )
    launcher = scope["launcher"]
    launcher._is_static = True
    launcher._npu_def_args = ["in_ptr0", "out_ptr0", "xnumel"]
    launcher._expected_positional_count = 3
    launcher.config = triton.Config({}, num_warps=4)
    launcher.cache_hash = "fast_test_kernel"
    launcher.n_regs = 0
    launcher.n_spills = 0
    launcher.shared = 0
    launcher.store_cubin = False
    launcher.runnable = True
    return launcher


@instantiate_parametrized_tests
class TestFastLauncherPython(TestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(isolated_fast_config(True))
        stack.enter_context(mock.patch(
            "torch._inductor.runtime.triton_heuristics.TritonBundler.put_winner"
        ))
        stack.enter_context(mock.patch.object(
            heuristics.CachingAutotuner, "run",
            side_effect=AssertionError("NPU fast dispatch must keep the NPU run lifecycle"),
        ))

    def test_build_clones_globals_and_retains_owner(self):
        kernel = FakeKernel()
        baseline = make_launcher(kernel)
        tuner = make_autotuner(baseline)
        fast_runner = mock.Mock()
        with mock.patch.object(
            torch_npu._C, "_FastNpuLauncher", return_value=fast_runner, create=True,
        ) as constructor:
            optimized = tuner._build_fast_launcher(baseline)
        self.assertIsNotNone(optimized, "NPU fast builder is not implemented")
        constructor.assert_called_once_with(kernel.loaded_kernel)
        self.assertIs(optimized.__code__, baseline.__code__)
        self.assertIsNot(optimized.__globals__, baseline.__globals__)
        self.assertIs(baseline.__globals__["runner"].__self__, kernel)
        self.assertIs(optimized.__globals__["runner"], fast_runner)
        self.assertIs(optimized._static_kernel_owner, kernel)
        for attr in (
            "config", "cache_hash", "n_regs", "n_spills", "shared",
            "store_cubin", "_is_static", "_npu_def_args", "_expected_positional_count",
        ):
            self.assertEqual(getattr(optimized, attr), getattr(baseline, attr))
        optimized("x", "out", 37, stream=11)
        fast_runner.assert_called_once_with(1, 1, 1, 11, "x", "out", 37)
        self.assertEqual(kernel.calls, [])
        baseline("old_x", "old_out", 19, stream=13)
        self.assertEqual(kernel.calls, [(1, 1, 1, 13, "old_x", "old_out", 19)])
        owner_ref = weakref.ref(kernel)
        del baseline, tuner, kernel
        gc.collect()
        self.assertIsNotNone(owner_ref())
        del optimized
        gc.collect()
        self.assertIsNone(owner_ref())

    @parametrize("enabled", [False, True])
    def test_unavailable_fast_uses_static_fallback(self, enabled):
        kernel = FakeKernel()
        baseline = make_launcher(kernel)
        tuner = make_autotuner(baseline, enabled=enabled)
        with mock.patch.object(
            torch_npu._C, "_FastNpuLauncher", create=True,
            side_effect=ValueError("unsupported fast schema"),
        ) as ctor:
            tuner.run("first", "out", 3, stream=0)
            tuner.run("second", "out", 5, stream=0)
        self.assertEqual(ctor.call_count, int(enabled))
        self.assertIs(tuner._cached_launcher, baseline)
        self.assertEqual(
            kernel.calls,
            [
                (1, 1, 1, 0, "first", "out", 3),
                (1, 1, 1, 0, "second", "out", 5),
            ],
        )

    def test_old_default_runner_is_not_silently_replaced(self):
        baseline = make_launcher(FakeKernel())
        scope = {"runner": baseline.__globals__["runner"]}
        exec("def old_launcher(stream, runner=runner):\n    runner(stream)\n", scope)
        old = scope["old_launcher"]
        old.__dict__.update(baseline.__dict__)
        with mock.patch.object(torch_npu._C, "_FastNpuLauncher", create=True) as ctor:
            self.assertIsNone(make_autotuner(old)._build_fast_launcher(old))
        ctor.assert_not_called()

    def test_cache_created_after_success_and_reused(self):
        baseline = make_launcher(FakeKernel())
        tuner = make_autotuner(baseline)
        events = []
        original = baseline.__globals__["runner"]
        baseline.__globals__["runner"] = lambda *args: (events.append("static"), original(*args))
        optimized = mock.Mock(side_effect=lambda *args, **kwargs: events.append("fast"))

        def build(winner):
            self.assertIs(winner, baseline)
            self.assertEqual(events, ["static"])
            events.append("build")
            return optimized

        with mock.patch.object(tuner, "_build_fast_launcher", side_effect=build) as builder:
            tuner.run("x", "out", 3, stream=0)
            tuner.run("next_x", "out", 9, stream=0)
            tuner.run("third_x", "out", 12, stream=0)
        self.assertEqual(events, ["static", "build", "fast", "fast"])
        builder.assert_called_once()
        self.assertIs(tuner.launchers[0], baseline)
        optimized.assert_any_call("next_x", "out", 9, stream=0)

    def test_release_benchmark_artifacts_drops_bound_run(self):
        baseline = make_launcher(FakeKernel())
        tuner = make_autotuner(baseline, enabled=False)
        tuner.run("x", "out", 3, stream=0)
        self.assertIn("run", tuner.__dict__)
        tuner.release_benchmark_artifacts()
        self.assertNotIn("run", tuner.__dict__)
        self.assertIsNone(tuner._cached_launcher)
        self.assertEqual(tuner.launchers, [])

    def test_failed_first_launch_does_not_cache(self):
        baseline = make_launcher(FakeKernel())
        baseline.__globals__["runner"] = mock.Mock(side_effect=RuntimeError("first failed"))
        tuner = make_autotuner(baseline)
        with mock.patch.object(tuner, "_build_fast_launcher") as builder:
            with self.assertRaisesRegex(RuntimeError, "first failed"):
                tuner.run("x", "out", 3, stream=0)
        builder.assert_not_called()
        self.assertIsNone(tuner._cached_launcher)

    def test_fast_launch_exception_is_not_replayed(self):
        kernel = FakeKernel()
        baseline = make_launcher(kernel)
        tuner = make_autotuner(baseline)
        optimized = mock.Mock(side_effect=RuntimeError("fast submit failed"))
        with mock.patch.object(tuner, "_build_fast_launcher", return_value=optimized):
            tuner.run("x", "out", 3, stream=0)
            with self.assertRaisesRegex(RuntimeError, "fast submit failed"):
                tuner.run("x", "out", 3, stream=0)
        optimized.assert_called_once()
        self.assertEqual(len(kernel.calls), 1)

    def test_winner_replacement_invalidates_fast(self):
        old = make_launcher(FakeKernel())
        new_kernel = FakeKernel()
        new = make_launcher(new_kernel)
        tuner = make_autotuner(old)
        optimized = mock.Mock()
        with mock.patch.object(tuner, "_build_fast_launcher", return_value=optimized) as builder:
            tuner.run("x", "out", 3, stream=0)
            tuner.run("x", "out", 3, stream=0)
            self.assertEqual(optimized.call_count, 1)
            tuner.launchers = [new]
            tuner.run("fresh", "out", 7, stream=0)
            self.assertEqual(optimized.call_count, 1)
            self.assertEqual(new_kernel.calls, [(1, 1, 1, 0, "fresh", "out", 7)])
            self.assertIs(builder.call_args.args[0], new)

    @parametrize("benchmark", [False, True])
    def test_special_call_bypasses_cached_fast_without_rebuilding(self, benchmark):
        kernel = FakeKernel()
        baseline = make_launcher(kernel)
        tuner = make_autotuner(baseline)
        optimized = mock.Mock()
        with mock.patch.object(
            tuner, "_build_fast_launcher", return_value=optimized,
        ) as builder:
            tuner.run("x", "out", 3, stream=0)
            tuner.run("x", "out", 3, stream=0)
            self.assertEqual(optimized.call_count, 1)
            if benchmark:
                tuner.run("special", "out", 7, stream=0, benchmark_run=True)
            else:
                tuner.run("special", "out", xnumel=7, stream=0)
            self.assertEqual(optimized.call_count, 1)
            self.assertEqual(kernel.calls[-1], (1, 1, 1, 0, "special", "out", 7))
            self.assertIs(tuner._cached_launcher, optimized)
            tuner.run("steady", "out", 9, stream=0)
            self.assertEqual(optimized.call_count, 2)
        builder.assert_called_once_with(baseline)

    def test_autotune_builds_only_for_selected_winner(self):
        loser = make_launcher(FakeKernel())
        winner_kernel = FakeKernel()
        winner = make_launcher(winner_kernel)
        tuner = make_autotuner(loser)
        tuner.launchers = [loser, winner]
        events = []

        def select(*args, **kwargs):
            # Controlled selection isolates the cache contract from timing noise.
            self.assertEqual(events, [])
            events.append("select")
            tuner.launchers = [winner]

        def build(launcher):
            self.assertIs(launcher, winner)
            self.assertEqual(len(winner_kernel.calls), 1)
            events.append("build")
            return mock.Mock()

        with (
            mock.patch.object(tuner, "autotune_to_one_config", side_effect=select),
            mock.patch.object(tuner, "_build_fast_launcher", side_effect=build) as builder,
        ):
            tuner.run("x", "out", 3, stream=0)
        builder.assert_called_once_with(winner)
        self.assertEqual(events, ["select", "build"])

    @parametrize("mode", ["input", "output", "input_output"])
    def test_downcast_wraps_optimized_launcher(self, mode):
        kernel = FakeKernel()
        baseline = make_launcher(kernel)
        cast_arg = "in_ptr0" if mode == "input" else "out_ptr0"
        mutated = {"out_ptr0"} if mode == "input_output" else set()
        wrapped = heuristics._wrap_launcher_with_downcast(
            baseline, {cast_arg: "*i64"}, mutated,
        )
        tuner = make_autotuner(wrapped)
        tuner.triton_meta["downcast_args"] = {cast_arg: "*i64"}
        tuner.mutated_arg_names = list(mutated)
        seen = []

        def run(grid0, grid1, grid2, stream, source, output, count):
            converted = source if mode == "input" else output
            self.assertEqual(converted.dtype, torch.int32)
            seen.append(True)
            if mode == "input_output":
                self.assertEqual(output, torch.tensor([7, 9], dtype=torch.int32))
                output.add_(1)
            else:
                output.copy_(source + 1)

        with mock.patch.object(torch_npu._C, "_FastNpuLauncher", return_value=run, create=True):
            optimized = tuner._build_fast_launcher(wrapped)
        self.assertIsNotNone(optimized)
        source = torch.tensor([2, 4], dtype=torch.int64 if mode == "input" else torch.int32)
        output = torch.tensor([7, 9], dtype=torch.int32 if mode == "input" else torch.int64)
        optimized(source, output, 2, stream=0)
        self.assertEqual(seen, [True])
        expected = [8, 10] if mode == "input_output" else [3, 5]
        self.assertEqual(output, torch.tensor(expected, dtype=output.dtype))


@instantiate_parametrized_tests
class TestFastLauncherDevice(TestCase):
    def setUp(self):
        super().setUp()
        if not torch.npu.is_available():
            self.skipTest("requires an available NPU")
        self.fast_type = require_fast_type()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(isolated_fast_config(True))

    def load(self, binary):
        interface = get_interface_for_device("npu")
        device = interface.current_device()
        kernel = NPUStaticArtifactAdapter.from_compiled_kernel(binary, {"device": device}, {})
        kernel.load_kernel(device)
        # Cleanup executes in reverse registration order: synchronize first.
        self.addCleanup(kernel.close)
        self.addCleanup(torch.npu.synchronize)
        return kernel, self.fast_type(kernel.loaded_kernel), interface.get_raw_stream(device)

    def test_multiple_tensors_and_none_pointer(self):
        @triton.jit(do_not_specialize=["read_optional"])
        def add(a, b, optional, out, read_optional: tl.int32):
            value = tl.load(a) + tl.load(b)
            if read_optional != 0:
                value += tl.load(optional)
            tl.store(out, value)

        a = torch.tensor([3], dtype=torch.int32, device="npu")
        b = torch.tensor([7], dtype=torch.int32, device="npu")
        out = torch.zeros_like(a)
        _, fast, stream = self.load(add.warmup(a, b, b, out, 3, grid=(1,)))
        fast(1, 1, 1, stream, a, b, None, out, 0)
        torch.npu.synchronize()
        self.assertEqual(out.cpu(), torch.tensor([10], dtype=torch.int32))
        fast(1, 1, 1, stream, a, b, b, out, 3)
        torch.npu.synchronize()
        self.assertEqual(out.cpu(), torch.tensor([17], dtype=torch.int32))

    def test_zero_runtime_arguments(self):
        @triton.jit
        def no_op():
            pass

        kernel, fast, stream = self.load(no_op.warmup(grid=(1,)))
        self.assertEqual(kernel.arg_kinds, ())
        fast(1, 1, 1, stream)
        torch.npu.synchronize()

    def test_signed_scalar_widths(self):
        @triton.jit(do_not_specialize=["a", "b", "c", "d"])
        def store(out, a: tl.int8, b: tl.int16, c: tl.int32, d: tl.int64):
            tl.store(out, a.to(tl.int64))
            tl.store(out + 1, b.to(tl.int64))
            tl.store(out + 2, c.to(tl.int64))
            tl.store(out + 3, d)

        out = torch.zeros(4, dtype=torch.int64, device="npu")
        values = (-113, -30001, -(2**30) + 7, -(2**40) + 19)
        kernel, fast, stream = self.load(store.warmup(out, *values, grid=(1,)))
        self.assertEqual(kernel.arg_kinds, ("tensor", "i8", "i16", "i32", "i64"))
        for current in (values, (127, 32767, 2**31 - 1, 2**63 - 1)):
            fast(1, 1, 1, stream, out, *current)
            torch.npu.synchronize()
            self.assertEqual(out.cpu(), torch.tensor(current, dtype=torch.int64))

    def test_unsigned_scalar_widths(self):
        @triton.jit(do_not_specialize=["a", "b", "c", "d"])
        def store(out, a: tl.uint8, b: tl.uint16, c: tl.uint32, d: tl.uint64):
            tl.store(out, a.to(tl.int64))
            tl.store(out + 1, b.to(tl.int64))
            tl.store(out + 2, c.to(tl.int64))
            tl.store(out + 3, d.to(tl.int64))

        out = torch.zeros(4, dtype=torch.int64, device="npu")
        values = (241, 60013, 2**31 + 17, 2**63 + 29)
        kernel, fast, stream = self.load(store.warmup(out, *values, grid=(1,)))
        self.assertEqual(kernel.arg_kinds, ("tensor", "u8", "u16", "u32", "u64"))
        fast(1, 1, 1, stream, out, *values)
        torch.npu.synchronize()
        self.assertEqual(out.cpu(), torch.tensor((*values[:3], 29 - 2**63), dtype=torch.int64))
        for index, bits in enumerate((8, 16, 32, 64)):
            for invalid in (-1, 2**bits):
                changed = list(values)
                changed[index] = invalid
                with self.assertRaises((TypeError, ValueError, RuntimeError, OverflowError)):
                    fast(1, 1, 1, stream, out, *changed)
                torch.npu.synchronize()
                self.assertEqual(out.cpu(), torch.tensor((*values[:3], 29 - 2**63), dtype=torch.int64))

    def test_bool_and_float_scalar_values(self):
        @triton.jit(do_not_specialize=["condition", "value"])
        def store(out, condition: tl.int1, value: tl.float32):
            tl.store(out, tl.where(condition, value, -value))

        out = torch.zeros(1, dtype=torch.float32, device="npu")
        kernel, fast, stream = self.load(store.warmup(out, True, 3.25, grid=(1,)))
        self.assertIn(kernel.arg_kinds, (
            ("tensor", "bool", "f32"),  # Triton 3.2: i1
            ("tensor", "u32", "f32"),  # Triton 3.6: u1
        ))
        for condition, value in ((True, 3.25), (False, 7.5), (True, -2.25)):
            fast(1, 1, 1, stream, out, condition, value)
            torch.npu.synchronize()
            expected = value if condition else -value
            self.assertEqual(out.cpu(), torch.tensor([expected], dtype=torch.float32))

    def test_argument_errors_do_not_execute(self):
        @triton.jit(do_not_specialize=["value"])
        def store(output, value: tl.int32):
            tl.store(output, value)

        output = torch.full((1,), 23, dtype=torch.int32, device="npu")
        _, fast, stream = self.load(store.warmup(output, 5, grid=(1,)))
        invalid = [
            (1, 1, 1, stream, output),
            (1, 1, 1, stream, output, 5, 7),
            (1, 1, 1, stream, object(), 5),
            (1, 1, 1, stream, output, 2**31),
            (1, 1, 1, stream, output, -(2**31) - 1),
            (0, 1, 1, stream, output, 5),
            (2**31, 1, 1, stream, output, 5),
        ]
        for args in invalid:
            with self.subTest(args=args):
                with self.assertRaises((TypeError, ValueError, RuntimeError, OverflowError)):
                    fast(*args)
                torch.npu.synchronize()
                self.assertEqual(output.cpu(), torch.tensor([23], dtype=torch.int32))

    @parametrize("enabled", [False, True])
    def test_compile_positive_hit_and_disable(self, enabled):
        with config.patch("use_fast_triton_launcher", enabled), observe_fast_builds() as records:
            compiled = torch.compile(
                lambda x, y: x + y, fullgraph=True,
                options={"npu_backend": "triton_experimental"},
            )
            x, y = torch.randn(1024, device="npu"), torch.randn(1024, device="npu")
            for _ in range(3):
                self.assertEqual(compiled(x, y), x + y)
        if enabled:
            self.assertTrue(any(record["fast"] is not None for record in records))
            self.assertGreater(sum(record["calls"] for record in records), 0)
        else:
            self.assertFalse(any(record["fast"] is not None for record in records))

    def test_dynamic_shapes_hit_fast(self):
        with observe_fast_builds() as records:
            compiled = torch.compile(
                lambda x: torch.relu(x + 1), dynamic=True, fullgraph=True,
                options={"npu_backend": "triton_experimental"},
            )
            for size in (257, 8193, 513, 257):
                x = torch.randn(size, device="npu")
                # A repeated call also exercises a newly specialized fast runner.
                for _ in range(2):
                    self.assertEqual(compiled(x), torch.relu(x + 1))
        self.assertGreater(sum(record["calls"] for record in records), 0)

    def test_nondefault_stream_and_graph_replay(self):
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), observe_fast_builds() as records:
            compiled = torch.compile(
                lambda x: x + 1, fullgraph=True,
                options={"npu_backend": "triton_experimental"},
            )
            x = torch.zeros(1024, device="npu")
            for _ in range(3):
                compiled(x)
            stream.synchronize()
            calls_before_capture = sum(record["calls"] for record in records)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                out = compiled(x)
            self.assertGreater(sum(record["calls"] for record in records), calls_before_capture)
            for value in (0.5, 1.0, -2.0):
                x.fill_(value)
                graph.replay()
                stream.synchronize()
                self.assertEqual(out.cpu(), torch.full((1024,), value + 1))


if __name__ == "__main__":
    run_tests()
