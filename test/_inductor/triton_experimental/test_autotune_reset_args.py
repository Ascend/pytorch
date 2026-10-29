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

        def bench(fn, device):
            for _ in range(3):
                fn()
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
            mock.patch.object(self.tuner, "_do_bench_npu", side_effect=lambda fn, device: fn()),
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
