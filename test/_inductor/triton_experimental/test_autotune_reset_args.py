# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch_npu._inductor.triton_experimental import npu_triton_heuristics as heuristics


@instantiate_parametrized_tests
class TestAutotuneResetArgs(TestCase):
    def setUp(self):
        super().setUp()
        self.tuner = object.__new__(heuristics.NPUCachingAutotuner)
        self.tuner.fn = SimpleNamespace(arg_names=["out"], __name__="test_reset")
        self.tuner.mutated_arg_names = ["out"]
        self.tuner.reset_to_zero_arg_names = ["out"]
        self.tuner.custom_kernel = False
        self.tuner.inductor_meta = {}
        self.tuner.coordesc_tuner = mock.Mock()
        self.device = mock.Mock()
        self.device.current_device.return_value = 0
        self.device.get_raw_stream.return_value = 0
        self.tuner.get_device_interface = mock.Mock(return_value=self.device)

    def make_launcher(self, observed):
        def launch(*args, out=None, stream=None):
            buffer = args[0] if args else out
            observed.append(buffer.item())
            buffer.add_(1)

        launcher = mock.Mock(side_effect=launch)
        launcher.config = SimpleNamespace(pre_hook=None, kwargs={})
        launcher.n_spills = 0
        launcher.runnable = True
        return launcher

    @parametrize("keyword", [False, True])
    @parametrize("mspti", [False, True])
    @parametrize("mutated", [False, True])
    def test_batch_resets_each_launched_buffer(self, keyword, mspti, mutated):
        self.tuner.mutated_arg_names = ["out"] if mutated else []
        observed = []
        self.tuner.launchers = [self.make_launcher(observed), self.make_launcher(observed)]
        original = torch.tensor([7.0])
        args, kwargs = ((), {"out": original}) if keyword else ((original,), {})

        def bench(fn, device, **options):
            for _ in range(6):
                options["prepare"]()
                fn()
            return 0.1

        def mspti_bench(launchers, run_one, warmup, active, device):
            for launcher in launchers:
                for _ in range(6):
                    run_one(launcher)
            return [0.1] * len(launchers)

        with (
            mock.patch.object(torch, "npu", create=True) as npu,
            mock.patch.object(heuristics, "_mspti_autotune_enabled", return_value=mspti),
            mock.patch.object(heuristics, "_warned_event_fallback", True),
            mock.patch.object(heuristics, "_mspti_bench_calls", side_effect=mspti_bench),
            mock.patch.object(self.tuner, "_do_bench_npu", side_effect=bench),
        ):
            timings = self.tuner._benchmark_all_configs_npu(*args, **kwargs)
        self.assertEqual(list(timings.values()), [0.1, 0.1])
        self.assertEqual(observed, [0.0] * 14)
        self.assertEqual(original, torch.tensor([7.0]))
        self.assertEqual(npu.synchronize.call_count, 2)

    @parametrize("keyword", [False, True])
    @parametrize("cpu_backup", [False, True])
    @parametrize("mutated", [False, True])
    def test_single_bench_resets_launcher_args(self, keyword, cpu_backup, mutated):
        self.tuner.mutated_arg_names = ["out"] if mutated else []
        original = torch.tensor([7.0])
        args, kwargs = ((), {"out": original}) if keyword else ((original,), {})
        copies = {"out": (original, original.clone())} if cpu_backup else {}
        observed = []
        launcher = self.make_launcher(observed)
        restore = self.tuner.restore_args_from_cpu

        def bench(fn, device, *, prepare, cleanup):
            for _ in range(3):
                prepare()
                fn()
                cleanup()
            return 0.1

        with (
            mock.patch.object(self.tuner, "copy_args_to_cpu_if_needed", return_value=copies),
            mock.patch.object(self.tuner, "restore_args_from_cpu", wraps=restore) as restore_mock,
            mock.patch.object(self.tuner, "_do_bench_npu", side_effect=bench),
        ):
            timing = self.tuner.bench(launcher, *args, **kwargs)
        self.assertEqual(timing, 0.1)
        self.assertEqual(observed, [0.0] * 3)
        self.assertEqual(original, torch.tensor([7.0]))
        self.assertEqual(restore_mock.call_count, 3)

    @parametrize("keyword", [False, True])
    def test_batch_without_reset_args_keeps_grouped_path(self, keyword):
        # No reset arguments must mean no prepare callback and no zeroing, so
        # the grouped-event path is reached exactly as before this change.
        self.tuner.reset_to_zero_arg_names = []
        observed = []
        self.tuner.launchers = [self.make_launcher(observed)]
        original = torch.tensor([7.0])
        args, kwargs = ((), {"out": original}) if keyword else ((original,), {})
        seen_options = []

        def bench(fn, device, **options):
            seen_options.append(options)
            fn()
            return 0.1

        with (
            mock.patch.object(torch, "npu", create=True),
            mock.patch.object(heuristics, "_mspti_autotune_enabled", return_value=False),
            mock.patch.object(heuristics, "_warned_event_fallback", True),
            mock.patch.object(self.tuner, "_do_bench_npu", side_effect=bench),
        ):
            timings = self.tuner._benchmark_all_configs_npu(*args, **kwargs)
        self.assertEqual(list(timings.values()), [0.1])
        self.assertEqual(seen_options, [{}])
        # The pre-run launch sees 7.0 and increments its own clone; the benched
        # clone is fresh, so it also sees 7.0 rather than a zeroed buffer.
        self.assertEqual(observed, [7.0, 7.0])
        self.assertEqual(original, torch.tensor([7.0]))

    def make_virtual_clock(self, setup_ms, kernel_ms, cleanup_ms=0.0):
        """Simulate host wall time and device event stamps from real costs.

        ``perf_counter`` advances only when prepare/launch/cleanup run, and
        event stamps are read off the same clock at ``record()``. The returned
        elapsed_time is therefore whatever the clock advanced between the two
        records, so a setup cost shows up in the sampling budget but can only
        appear in the reported timing if the event window actually covers it.
        """
        clock = SimpleNamespace(now=0.0, launches=0, events=0, trace=[])

        class Event:
            def __init__(self, index, **kwargs):
                self.index = index
                self.stamp = 0.0

            def record(self):
                self.stamp = clock.now
                clock.trace.append(self.index)

            def elapsed_time(self, end):
                return end.stamp - self.stamp

        def make_event(**kwargs):
            event = Event(clock.events, **kwargs)
            clock.events += 1
            return event

        self.device.Event = make_event

        def prepare():
            clock.trace.append("prepare")
            clock.now += setup_ms

        def launch():
            clock.trace.append("launch")
            clock.now += kernel_ms[clock.launches % len(kernel_ms)]
            clock.launches += 1

        def cleanup():
            clock.trace.append("cleanup")
            clock.now += cleanup_ms

        clock.perf_counter = lambda: clock.now / 1000.0
        clock.prepare = prepare
        clock.launch = launch
        clock.cleanup = cleanup
        return clock

    def run_virtual_bench(self, clock, **options):
        cache = mock.Mock()
        cache.zero_.side_effect = lambda: clock.trace.append("flush")
        with (
            mock.patch.object(torch, "empty", return_value=cache),
            mock.patch.object(heuristics.time, "perf_counter", clock.perf_counter),
        ):
            return self.tuner._do_bench_npu(
                clock.launch, self.device,
                prepare=clock.prepare, cleanup=clock.cleanup, **options
            )

    @parametrize(
        "setup_ms,kernel_ms,launches",
        [
            # Slow setup, fast kernel: the budget collapses to the 3-sample
            # floor even though the measured kernel is cheap.
            (40.0, 0.5, 5),
            # Fast setup, slow kernel: the same budget for the opposite split,
            # so a single wall-clock sample cannot tell the two apart.
            (0.25, 30.0, 5),
            # Both cheap: the full 10 warmup + 20 measured budget.
            (0.25, 0.5, 32),
            # Setup dominates the kernel but is still cheap enough for the
            # full budget.
            (2.0, 0.25, 32),
        ],
    )
    def test_setup_cost_drives_budget_but_not_reported_time(self, setup_ms, kernel_ms, launches):
        clock = self.make_virtual_clock(setup_ms, [kernel_ms])
        timing = self.run_virtual_bench(clock)
        # Only the kernel is inside the event window, whatever setup costs.
        self.assertEqual(timing, kernel_ms)
        self.assertEqual(clock.launches, launches)

    def test_estimate_sample_warms_up_before_measurement(self):
        # A single setup+launch above the whole warmup budget drives
        # warmup_count to 0; the untimed call and the estimate are still two
        # full warm launches ahead of the first measured window.
        clock = self.make_virtual_clock(40.0, [0.5])
        with mock.patch.object(heuristics.ncfg, "event_bench_max_samples", 3):
            timing = self.run_virtual_bench(clock)
        self.assertEqual(timing, 0.5)
        self.assertEqual(clock.launches, 5)
        # Event pairs are (0, 1), (2, 3), (4, 5); the measured loop is the
        # first window opened by pair 1, i.e. event index 2.
        measured_start = clock.trace.index(2, 6)
        self.assertEqual(clock.trace[:2], ["prepare", "launch"])
        self.assertEqual(clock.trace[measured_start - 2:measured_start], ["flush", "prepare"])
        self.assertEqual(clock.trace[:measured_start].count("launch"), 3)
        self.assertEqual(clock.trace.count("flush"), 4)

    @parametrize("phase", list(range(10)))
    def test_median_keeps_candidate_ranking_stable(self, phase):
        # A steady 1.0ms candidate with a 4.0ms outlier every tenth launch must
        # still rank ahead of a steady 1.5ms candidate, whatever launch the
        # budget estimate lands on.
        pattern = [1.0] * 9 + [4.0]
        jittery = pattern[phase:] + pattern[:phase]
        fast = self.make_virtual_clock(0.25, jittery)
        fast_timing = self.run_virtual_bench(fast)
        steady = self.make_virtual_clock(0.25, [1.5])
        steady_timing = self.run_virtual_bench(steady)
        self.assertEqual(fast_timing, 1.0)
        self.assertEqual(steady_timing, 1.5)
        self.assertLess(fast_timing, steady_timing)

    @parametrize("sample_limit", [1, 2, 3, 4, 20, 40])
    def test_sample_count_follows_config_limit(self, sample_limit):
        clock = self.make_virtual_clock(0.25, [0.5])
        with mock.patch.object(heuristics.ncfg, "event_bench_max_samples", sample_limit):
            timing = self.run_virtual_bench(clock)
        self.assertEqual(timing, 0.5)
        self.assertEqual(clock.events, 2 * sample_limit)
        # 1 untimed + 1 estimate + 10 warmup + the configured measured samples,
        # which the rep budget allows in full at this cost. The old hardcoded
        # 20 was both the default and the ceiling; now only the default.
        self.assertEqual(clock.launches, 12 + sample_limit)

    @parametrize("sample_limit", [0, -5])
    def test_illegal_sample_limit_clamps_to_one_pair(self, sample_limit):
        clock = self.make_virtual_clock(0.25, [0.5])
        with mock.patch.object(heuristics.ncfg, "event_bench_max_samples", sample_limit):
            timing = self.run_virtual_bench(clock)
        self.assertEqual(timing, 0.5)
        self.assertEqual(clock.events, 2)
        self.assertEqual(clock.launches, 13)

    def test_cleanup_error_closes_timing_window_first(self):
        clock = self.make_virtual_clock(0.25, [0.5])
        original_cleanup = clock.cleanup
        calls = []

        def cleanup():
            original_cleanup()
            calls.append(len(calls) + 1)
            if len(calls) == 2:
                raise RuntimeError("cleanup failed")

        clock.cleanup = cleanup
        with (
            mock.patch.object(heuristics.ncfg, "event_bench_max_samples", 3),
            self.assertRaisesRegex(RuntimeError, "cleanup failed"),
        ):
            self.run_virtual_bench(clock)
        # The failing cleanup is the estimate sample's; its end event was
        # already recorded, so the timing window closed before unwinding.
        self.assertEqual(clock.trace[-5:], ["prepare", 0, "launch", 1, "cleanup"])

    def test_event_pairs_exclude_setup_and_cleanup(self):
        operations = []
        events = []

        class Event:
            def __init__(self, **kwargs):
                self.index = len(events)
                events.append(self)

            def record(self):
                operations.append(self.index)

            def elapsed_time(self, end):
                return [9.0, 1.0, 4.0][self.index // 2 % 3]

        self.device.Event = Event
        cache = mock.Mock()
        cache.zero_.side_effect = lambda: operations.append("flush")
        with (
            mock.patch.object(torch, "empty", return_value=cache),
            mock.patch.object(heuristics.ncfg, "event_bench_max_inner", 1),
            mock.patch.object(heuristics.ncfg, "event_bench_max_groups", 3),
            mock.patch.object(heuristics.ncfg, "event_bench_target_ms", 1),
        ):
            timing = self.tuner._do_bench_npu(
                lambda: operations.append("launch"),
                self.device,
                warmup=0,
                rep=0,
                prepare=lambda: operations.append("reset"),
                cleanup=lambda: operations.append("restore"),
            )
        self.assertEqual(timing, 4.0)
        self.assertEqual(len(events), 40)
        self.assertEqual(operations[:3], ["reset", "launch", "restore"])
        self.assertEqual(operations[3:43], list(range(40)))
        operations = operations[43:]
        for i, operation in enumerate(operations):
            if isinstance(operation, int) and operation % 2 == 0:
                self.assertGreaterEqual(i, 2)
                self.assertEqual(operations[i - 2:i + 4], ["flush", "reset", operation, "launch", operation + 1, "restore"])
        self.assertEqual(self.device.synchronize.call_count, 5)

    @parametrize("sample_seconds", [0.001, 0.05])
    def test_reset_sampling_has_bounded_default_budget(self, sample_seconds):
        launch = mock.Mock()
        prepare = mock.Mock()
        cache = mock.Mock()
        self.device.Event.return_value.elapsed_time.return_value = 1.0
        with (
            mock.patch.object(torch, "empty", return_value=cache),
            mock.patch.object(heuristics.ncfg, "event_bench_max_inner", 256),
            mock.patch.object(heuristics.time, "perf_counter", side_effect=[0.0, sample_seconds]),
        ):
            timing = self.tuner._do_bench_npu(launch, self.device, prepare=prepare)
        self.assertEqual(timing, 1.0)
        expected_launches = 32 if sample_seconds == 0.001 else 5
        self.assertEqual(launch.call_count, expected_launches)
        self.assertEqual(prepare.call_count, expected_launches)
        self.assertEqual(cache.zero_.call_count, expected_launches - 1)
        self.assertEqual(self.device.Event.call_count, 40)

    def test_single_bench_restores_args_after_launch_error(self):
        restore = mock.Mock()
        launcher = self.make_launcher([])
        launcher.side_effect = RuntimeError("launch failed")
        with (
            mock.patch.object(self.tuner, "copy_args_to_cpu_if_needed", return_value={}),
            mock.patch.object(self.tuner, "restore_args_from_cpu", restore),
            self.assertRaisesRegex(RuntimeError, "launch failed"),
        ):
            self.tuner.bench(launcher, torch.tensor([7.0]))
        restore.assert_called_once()

    @parametrize("failure_call", [1, 2, 3, 4])
    @parametrize("warmup", [0, 25])
    def test_prepare_error_runs_cleanup(self, failure_call, warmup):
        active = False
        calls = 0
        cleanups = 0

        def prepare():
            nonlocal active, calls
            active = True
            calls += 1
            if calls == failure_call:
                raise RuntimeError("prepare failed")

        def cleanup():
            nonlocal active, cleanups
            active = False
            cleanups += 1

        self.device.Event.return_value.elapsed_time.return_value = 1.0
        with (
            mock.patch.object(torch, "empty", return_value=mock.Mock()),
            mock.patch.object(heuristics.time, "perf_counter", side_effect=[0.0, 0.001]),
            self.assertRaisesRegex(RuntimeError, "prepare failed"),
        ):
            self.tuner._do_bench_npu(
                lambda: None, self.device, warmup=warmup, rep=0,
                prepare=prepare, cleanup=cleanup,
            )
        self.assertFalse(active)
        self.assertEqual(cleanups, failure_call)

    def test_without_setup_uses_grouped_events(self):
        launch = mock.Mock()
        self.device.Event.return_value.elapsed_time.return_value = 1.0
        with (
            mock.patch.object(torch, "empty", return_value=mock.Mock()),
            mock.patch.object(heuristics.ncfg, "event_bench_max_inner", 1),
            mock.patch.object(heuristics.ncfg, "event_bench_max_groups", 3),
            mock.patch.object(heuristics.ncfg, "event_bench_target_ms", 1),
        ):
            timing = self.tuner._do_bench_npu(launch, self.device, warmup=0, rep=0)
        self.assertEqual(timing, 1.0)
        self.assertEqual(launch.call_count, 15)
        self.assertEqual(self.device.Event.call_count, 8)
        self.assertEqual(self.device.synchronize.call_count, 6)

    @parametrize("stage", ["reset", "launch"])
    def test_single_bench_restores_after_error(self, stage):
        original = torch.tensor([7.0])
        copies = {"out": (original, original.clone())}
        launcher = self.make_launcher([])
        reset = self.tuner.reset_to_zero_args

        def reset_or_fail(*args, **kwargs):
            reset(*args, **kwargs)
            if stage == "reset":
                raise RuntimeError("injected failure")

        if stage == "launch":
            launcher.side_effect = RuntimeError("injected failure")
        with (
            mock.patch.object(self.tuner, "copy_args_to_cpu_if_needed", return_value=copies),
            mock.patch.object(self.tuner, "reset_to_zero_args", side_effect=reset_or_fail),
            self.assertRaisesRegex(RuntimeError, "injected failure"),
        ):
            self.tuner.bench(launcher, original)
        self.assertEqual(original, torch.tensor([7.0]))


    def test_non_reset_bench_restores_after_launch_error(self):
        self.tuner.reset_to_zero_arg_names = []
        original = torch.tensor([7.0])
        copies = {"out": (original, original.clone())}
        launcher = self.make_launcher([])

        def fail(*args, **kwargs):
            original.add_(1)
            raise RuntimeError("launch failed")

        launcher.side_effect = fail
        with (
            mock.patch.object(self.tuner, "copy_args_to_cpu_if_needed", return_value=copies),
            self.assertRaisesRegex(RuntimeError, "launch failed"),
        ):
            self.tuner.bench(launcher, original)
        self.assertEqual(original, torch.tensor([7.0]))

    def test_without_reset_preserves_launch_input(self):
        self.tuner.reset_to_zero_arg_names = []
        original = torch.tensor([7.0])
        observed = []
        with (
            mock.patch.object(self.tuner, "copy_args_to_cpu_if_needed", return_value={}),
            mock.patch.object(self.tuner, "_do_bench_npu", side_effect=lambda fn, device: fn()),
        ):
            self.tuner.bench(self.make_launcher(observed), original)
        self.assertEqual(observed, [7.0])
        self.assertEqual(original, torch.tensor([7.0]))


if __name__ == "__main__":
    run_tests()
