"""Autotune must not open a profiler inside one the caller already opened.

The torch_npu profiler backend is a process-wide singleton: a session started
inside another one finalizes the shared trace when it exits and clears
ProfPathCreator, so the outer session is torn down early and reports "Incorrect
schedule: Stop profiler while current state is RECORD". With aggresive_autotune
on, both the batch benchmark and the bandwidth benchmark would open one, so each
has to probe the global state and fall back to the event timer instead.
"""

from types import SimpleNamespace
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)

import torch_npu
import torch_npu._inductor  # noqa: F401
from torch_npu._inductor.runtime import triton_heuristics


class TestAutotuneProfilerSession(TestCase):
    @staticmethod
    def _autotuner(profile_bandwidth=True):
        """An autotuner carrying just enough state to reach the benchmark call."""
        autotuner = object.__new__(triton_heuristics.NPUCachingAutotuner)
        autotuner.inductor_meta = {
            "profile_bandwidth_with_do_bench_using_profiling": profile_bandwidth,
        }
        autotuner.get_device_interface = lambda: SimpleNamespace(
            current_device=lambda: 0,
            get_raw_stream=lambda _device: None,
        )
        autotuner.clone_args = lambda *args, **kwargs: ((), {})
        autotuner.reset_to_zero_args = lambda *args, **kwargs: None
        return autotuner

    @parametrize("is_prof_inited", (True, False))
    def test_probe_follows_the_global_profiler_state(self, is_prof_inited):
        from torch_npu.profiler._profiler_path_creator import ProfPathCreator

        with patch.object(ProfPathCreator(), "is_prof_inited", is_prof_inited):
            self.assertEqual(
                triton_heuristics.npu_profiler_session_active(),
                is_prof_inited,
            )

    def test_probe_never_breaks_a_kernel_launch(self):
        # The probe runs on the launch path, so a profiler backend it cannot
        # reach has to read as "no session" rather than raise.
        import torch_npu.profiler._profiler_path_creator as path_creator

        with patch.object(
            path_creator,
            "ProfPathCreator",
            side_effect=RuntimeError("no profiler backend"),
        ):
            self.assertFalse(triton_heuristics.npu_profiler_session_active())

    @parametrize("session_active", (True, False))
    def test_batch_benchmark_is_skipped_while_the_caller_profiles(
        self, session_active
    ):
        autotuner = object.__new__(triton_heuristics.NPUCachingAutotuner)
        kernel_funcs = [lambda: None]

        with (
            patch.object(triton_heuristics.npu_config, "aggresive_autotune", True),
            patch.object(
                triton_heuristics,
                "npu_profiler_session_active",
                return_value=session_active,
            ),
            patch.object(
                triton_heuristics,
                "mspti_batch_benchmark",
                return_value=[1.0],
            ) as batch_benchmark,
        ):
            timings = autotuner._benchmark_kernel_funcs_batch(kernel_funcs, "grouped")

        if session_active:
            # None sends the caller back to the per-config timer.
            self.assertIsNone(timings)
            batch_benchmark.assert_not_called()
        else:
            self.assertEqual(timings, (1.0,))
            batch_benchmark.assert_called_once()

    @parametrize("session_active", (True, False))
    def test_bandwidth_benchmark_falls_back_to_the_timer(self, session_active):
        autotuner = self._autotuner()

        with (
            patch.object(
                triton_heuristics,
                "npu_profiler_session_active",
                return_value=session_active,
            ),
            patch.object(
                triton_heuristics,
                "do_bench_using_profiling_npu",
                return_value=2.5,
            ) as profiling_benchmark,
            patch.object(
                triton_heuristics.benchmarker,
                "benchmark_gpu",
                return_value=1.5,
            ) as event_timer,
        ):
            timing = autotuner._bench_with_launch_args(lambda **kwargs: None, (), ())

        if session_active:
            self.assertEqual(timing, 1.5)
            profiling_benchmark.assert_not_called()
            event_timer.assert_called_once()
        else:
            self.assertEqual(timing, 2.5)
            profiling_benchmark.assert_called_once()
            event_timer.assert_not_called()

    def test_probe_stays_behind_the_bandwidth_config_check(self):
        autotuner = self._autotuner(profile_bandwidth=False)

        with (
            patch.object(
                triton_heuristics,
                "npu_profiler_session_active",
                return_value=False,
            ) as probe,
            patch.object(
                triton_heuristics.benchmarker,
                "benchmark_gpu",
                return_value=1.5,
            ),
        ):
            timing = autotuner._bench_with_launch_args(lambda **kwargs: None, (), ())

        self.assertEqual(timing, 1.5)
        # The probe imports from torch_npu.profiler, and autotune benchmarks
        # every config, so it must not run when there is nothing to guard.
        probe.assert_not_called()


instantiate_parametrized_tests(TestAutotuneProfilerSession)


if __name__ == "__main__":
    run_tests()
