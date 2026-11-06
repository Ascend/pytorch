import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch._inductor.runtime.triton_heuristics import Config, HeuristicType, hash_configs

from torch_npu._inductor import config as npu_config
from torch_npu._inductor.runtime import triton_heuristics
from torch_npu._inductor.runtime.triton_heuristics import (
    NPUCachingAutotuner,
    NPUSymbolicGroupedAutotuner,
)


class TestAutotuneSwitch(TestCase):
    def test_backend_import_preserves_upstream_max_flags(self):
        env = os.environ.copy()
        for name in ("TORCHINDUCTOR_MAX_AUTOTUNE", "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"):
            env.pop(name, None)
        for max_autotune, max_pointwise in (
            (False, False), (False, True), (True, False), (True, True)
        ):
            with self.subTest(flags=(max_autotune, max_pointwise)):
                code = (
                    "import json; from torch._inductor import config; "
                    f"config.max_autotune={max_autotune!r}; "
                    f"config.max_autotune_pointwise={max_pointwise!r}; "
                    "import torch_npu._inductor; "
                    "print(json.dumps([config.max_autotune, config.max_autotune_pointwise]))"
                )
                output = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
                self.assertEqual(json.loads(output.splitlines()[-1]), [max_autotune, max_pointwise])

    def _autotuner(self, heuristic_type, inductor_meta):
        tuner = object.__new__(NPUCachingAutotuner)
        first_launcher = SimpleNamespace(config=object())
        second_launcher = SimpleNamespace(config=object())
        first_candidate = {"candidate_id": 0, "variant_id": "first"}
        second_candidate = {"candidate_id": 1, "variant_id": "second"}
        tuner.candidate_plan = {
            "candidate_entries": [first_candidate, second_candidate]
        }
        tuner.compiled_candidate_entries = (
            {"candidate": first_candidate, "launcher": first_launcher},
            {"candidate": second_candidate, "launcher": second_launcher},
        )
        tuner.variant_launcher_map = {"first": first_launcher, "second": second_launcher}
        tuner.launchers = [first_launcher, second_launcher]
        tuner.heuristic_type = heuristic_type
        tuner.inductor_meta = inductor_meta
        tuner.save_cache_hook = mock.Mock()
        tuner.autotune_to_one_config = mock.Mock()
        tuner.get_fn_name = lambda: "autotune_switch_test"
        tuner.best_candidate_config = None
        tuner.best_launcher = None
        tuner.best_runtime_blocks = ()
        tuner.precompile_time_taken_ns = 0
        return tuner, first_launcher

    def test_max_off_still_benchmarks_existing_candidates(self):
        for heuristic_type in (
            HeuristicType.POINTWISE,
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
        ):
            with self.subTest(heuristic_type=heuristic_type):
                tuner, _ = self._autotuner(heuristic_type, {
                    "max_autotune": False, "max_autotune_pointwise": False,
                })
                tuner.autotuner(stream=0)
                tuner.autotune_to_one_config.assert_called_once_with()
                tuner.save_cache_hook.assert_not_called()

    def test_explicit_max_environment_off_still_allows_benchmarking(self):
        for name in ("TORCHINDUCTOR_MAX_AUTOTUNE", "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE"):
            with self.subTest(name=name), mock.patch.dict(os.environ, {name: "0"}):
                tuner, _ = self._autotuner(HeuristicType.POINTWISE, {
                    "max_autotune": False, "max_autotune_pointwise": False,
                })
                tuner.autotuner(stream=0)
                tuner.autotune_to_one_config.assert_called_once_with()

    def test_explicit_pointwise_max_switch_enables_benchmarking(self):
        tuner, _ = self._autotuner(
            HeuristicType.POINTWISE,
            {"max_autotune": False, "max_autotune_pointwise": True},
        )
        with mock.patch.object(tuner, "autotune_to_one_config") as benchmark:
            tuner.autotuner(stream=0)
        benchmark.assert_called_once_with()

    def test_deterministic_reduction_skips_benchmark(self):
        tuner, first_launcher = self._autotuner(
            HeuristicType.REDUCTION,
            {
                "max_autotune": True,
                "deterministic": True,
            },
        )
        tuner.autotuner(stream=0)
        self.assertEqual(tuner.launchers, [first_launcher])
        tuner.autotune_to_one_config.assert_not_called()

    def _grouped_autotuner(self, heuristic_type, inductor_meta):
        tuner = object.__new__(NPUSymbolicGroupedAutotuner)
        first = {"candidate_id": "g0_c0", "variant_id": "v0", "full_config": {}}
        second = {"candidate_id": "g0_c1", "variant_id": "v1", "full_config": {}}
        first_launcher = SimpleNamespace(config=object(), runnable=True)
        second_launcher = SimpleNamespace(config=object(), runnable=True)
        third = dict(first, candidate_id="g2_c0")
        fourth = dict(second, candidate_id="g2_c1")
        third_launcher = SimpleNamespace(config=object(), runnable=True)
        fourth_launcher = SimpleNamespace(config=object(), runnable=True)
        tuner.candidate_plan = {"group_to_candidates": [
            [first, second], [], [third, fourth],
        ]}
        tuner.reachable_selection_keys = (0, 2)
        tuner.compiled_candidate_entries = (
            {"candidate": first, "launcher": first_launcher},
            {"candidate": second, "launcher": second_launcher},
            {"candidate": third, "launcher": third_launcher},
            {"candidate": fourth, "launcher": fourth_launcher},
        )
        tuner.best_candidate_map = {}
        tuner.best_launcher_map = {}
        tuner.inductor_meta = inductor_meta
        tuner.heuristic_type = heuristic_type
        tuner.lock = threading.Lock()
        tuner.get_fn_name = lambda: "grouped_autotune_switch_test"
        tuner._precompile_variants = mock.Mock()
        tuner._autotune_all_groups = mock.Mock()

        return tuner, first, first_launcher

    def test_grouped_max_off_still_benchmarks_existing_candidates(self):
        for heuristic_type in (
            HeuristicType.POINTWISE,
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
        ):
            with self.subTest(heuristic_type=heuristic_type):
                tuner, _, _ = self._grouped_autotuner(heuristic_type, {
                    "max_autotune": False, "max_autotune_pointwise": False,
                })
                tuner.ensure_grouped_autotune_ready()
                tuner._autotune_all_groups.assert_called_once_with()
                self.assertEqual(tuner.best_candidate_map, {})

    def test_grouped_deterministic_reduction_picks_first_per_group(self):
        for heuristic_type in (HeuristicType.REDUCTION, HeuristicType.PERSISTENT_REDUCTION):
            with self.subTest(heuristic_type=heuristic_type):
                tuner, first, first_launcher = self._grouped_autotuner(heuristic_type, {
                    "max_autotune": False, "max_autotune_pointwise": False, "deterministic": True,
                })
                tuner.ensure_grouped_autotune_ready()
                tuner._autotune_all_groups.assert_not_called()
                self.assertIs(tuner.best_candidate_map[0], first)
                self.assertIs(tuner.best_launcher_map[0], first_launcher)
                self.assertIs(
                    tuner.best_candidate_map[2],
                    tuner.candidate_plan["group_to_candidates"][2][0],
                )
                self.assertNotIn(1, tuner.best_candidate_map)

    def test_grouped_deterministic_reduction_skips_unrunnable_candidates(self):
        tuner, _, launcher = self._grouped_autotuner(
            HeuristicType.REDUCTION, {"deterministic": True}
        )
        launcher.runnable = False
        tuner.ensure_grouped_autotune_ready()
        self.assertIs(
            tuner.best_candidate_map[0],
            tuner.candidate_plan["group_to_candidates"][0][1],
        )
        tuner._autotune_all_groups.assert_not_called()

    def test_grouped_deterministic_pointwise_still_benchmarks(self):
        tuner, _, _ = self._grouped_autotuner(HeuristicType.POINTWISE, {"deterministic": True})
        tuner.ensure_grouped_autotune_ready()
        tuner._autotune_all_groups.assert_called_once_with()

    def test_single_candidate_does_not_benchmark(self):
        tuner, first_launcher = self._autotuner(HeuristicType.POINTWISE, {})
        tuner.candidate_plan["candidate_entries"] = tuner.candidate_plan["candidate_entries"][:1]
        tuner.compiled_candidate_entries = tuner.compiled_candidate_entries[:1]
        tuner.launchers = [first_launcher]
        tuner.autotuner(stream=0)
        self.assertIs(tuner.best_launcher, first_launcher)
        tuner.autotune_to_one_config.assert_not_called()

    def test_selected_candidate_does_not_benchmark_again(self):
        tuner, first_launcher = self._autotuner(HeuristicType.POINTWISE, {})
        tuner.best_launcher = first_launcher
        tuner.autotuner(stream=0)
        tuner.autotune_to_one_config.assert_not_called()

    def test_determinism_overrides_max_flags_only_for_reductions(self):
        for heuristic_type in (
            HeuristicType.POINTWISE,
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
        ):
            for flag in ("deterministic", "are_deterministic_algorithms_enabled"):
                for max_autotune, max_pointwise in (
                    (False, False), (False, True), (True, False), (True, True)
                ):
                    with self.subTest(
                        heuristic_type=heuristic_type, flag=flag,
                        flags=(max_autotune, max_pointwise),
                    ):
                        tuner, first = self._autotuner(heuristic_type, {
                            "max_autotune": max_autotune,
                            "max_autotune_pointwise": max_pointwise, flag: True,
                        })
                        tuner.autotuner(stream=0)
                        if heuristic_type == HeuristicType.POINTWISE:
                            tuner.autotune_to_one_config.assert_called_once_with()
                        else:
                            tuner.autotune_to_one_config.assert_not_called()
                            self.assertIs(tuner.best_launcher, first)
                            tuner.save_cache_hook.assert_called_once_with(
                                first.config, 0, found_by_coordesc=False,
                                triton_cache_hash=None,
                            )

    def test_deterministic_selection_preserves_upstream_cache_metadata(self):
        for coordinate_descent in (False, True):
            with self.subTest(coordinate_descent=coordinate_descent):
                tuner, first = self._autotuner(HeuristicType.REDUCTION, {
                    "deterministic": True,
                    "coordinate_descent_tuning": coordinate_descent,
                })
                first.cache_hash = "selected_triton_cache_key"
                full_config = triton_heuristics.config_to_dict(Config({"X0BLOCK": 64}))
                tuner.candidate_plan["candidate_entries"][0]["full_config"] = full_config
                tuner.autotuner(stream=0)
                tuner.autotune_to_one_config.assert_not_called()
                self.assertEqual(tuner.best_candidate_config.kwargs, {"X0BLOCK": 64})
                tuner.save_cache_hook.assert_called_once_with(
                    tuner.best_candidate_config, 0, found_by_coordesc=coordinate_descent,
                    triton_cache_hash=first.cache_hash,
                )

    def test_pointwise_forwards_candidate_generation_result(self):
        configs = [object(), object()]
        for max_autotune, max_pointwise, autotune_pointwise in (
            (False, False, True), (False, False, False),
            (True, False, False), (False, True, False),
        ):
            with self.subTest(
                flags=(max_autotune, max_pointwise, autotune_pointwise)
            ), mock.patch.object(
                triton_heuristics, "triton_config_npu_index", return_value=configs
            ), mock.patch.object(triton_heuristics, "cached_autotune") as cached:
                triton_heuristics.pointwise({"x0": 1024}, {}, inductor_meta={
                    "max_autotune": max_autotune, "max_autotune_pointwise": max_pointwise,
                    "autotune_pointwise": autotune_pointwise,
                })
                self.assertEqual(cached.call_args.args[1], configs)

    def test_autotune_pointwise_does_not_trim_reduction_candidates(self):
        configs = [object(), object()]
        for generate in (triton_heuristics.reduction, triton_heuristics.persistent_reduction):
            with self.subTest(heuristic=generate.__name__), mock.patch.object(
                triton_heuristics, "triton_config_npu_index", return_value=configs
            ), mock.patch.object(triton_heuristics, "cached_autotune") as cached:
                generate({"x0": 64, "r0_": 1024}, triton_meta={}, inductor_meta={
                    "max_autotune": False, "max_autotune_pointwise": False,
                    "autotune_pointwise": False,
                })
                self.assertEqual(cached.call_args.args[1], configs)


class TestAutotuneCandidateGeneration(TestCase):
    def setUp(self):
        super().setUp()
        for name, value in (
            ("num_vector_core", 32), ("ub_size", 192 * 1024),
            ("fasta_autotune", False), ("enable_costmodel_prefilter", False),
        ):
            patcher = mock.patch.object(npu_config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ, {"FAST_RUN_WITH_MAX_TILING_NUM": "-1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _key(cfg):
        return (tuple(sorted(cfg.kwargs.items())), cfg.num_warps, cfg.num_stages)

    def _generate(self, kind="pointwise", flags=(False, False),
                  kernel_type="simd", grouped=False, autotune_pointwise=True, **extra):
        shape = {"x0": 4096} if kind == "pointwise" else {"x0": 128, "r0_": 1024}
        meta = {
            "kernel_name": "candidate_switch_test", "axis_names": tuple(shape),
            "split_axis": [0], "tiling_axis": list(range(len(shape))),
            "no_loop_axis": [], "low_dims": [len(shape) - 1],
            "split_axis_dtype": torch.float32, "dual_reduction": False,
            "npu_kernel_type": kernel_type, "max_autotune": flags[0],
            "max_autotune_pointwise": flags[1],
            "autotune_pointwise": autotune_pointwise,
        }
        if grouped:
            meta.update({
                "group_enabled": True, "primary_group_axis": "x0",
                "runtime_block_arg_names": ("X0BLOCK",),
                "axis_static_values": () if kind == "pointwise" else (("r0_", 1024),),
                "group_features": ({
                    "name": "pointwise" if kind == "pointwise" else "outer",
                    "source": "primary_axis", "axis_names": ("x0",),
                    "buckets": (128, 4096),
                },),
            })
        meta.update(extra)
        signature = {"in_ptr0": "*fp32"}
        signature.update({f"{axis}_numel": "i32" for axis in shape})
        configs = triton_heuristics.triton_config_npu_index(
            shape, meta, triton_meta={"signature": signature},
            is_reduction=kind == "reduction",
            is_persistent_reduction=kind == "persistent",
        )
        return configs, meta

    def test_max_flags_preserve_existing_candidate_sets(self):
        for kind in ("pointwise", "reduction", "persistent"):
            for kernel_type in ("simd", "simt_only", "simt_template", "simd_simt_mix"):
                with self.subTest(kind=kind, kernel_type=kernel_type):
                    basic, _ = self._generate(kind, kernel_type=kernel_type)
                    basic_keys = [self._key(cfg) for cfg in basic]
                    self.assertGreater(len(basic_keys), 1)
                    for flags in ((False, True), (True, False), (True, True)):
                        configs, _ = self._generate(kind, flags, kernel_type)
                        self.assertEqual(basic_keys, [self._key(cfg) for cfg in configs])
                        self.assertEqual(hash_configs(basic), hash_configs(configs))
                    if kernel_type == "simt_only":
                        self.assertEqual({cfg.num_warps for cfg in basic}, {8, 16, 32, 64})
                    else:
                        self.assertEqual({cfg.kwargs["multibuffer"] for cfg in basic
                                          if cfg.kwargs["compile_mode"] != "simt_only"},
                                         {False, True})

    def test_max_environment_and_gemm_flags_preserve_existing_candidates(self):
        for kind in ("pointwise", "reduction", "persistent"):
            with self.subTest(kind=kind):
                basic, _ = self._generate(kind)
                with mock.patch.dict(os.environ, {
                    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
                    "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "1",
                }):
                    same, _ = self._generate(kind, max_autotune_gemm=True)
                    missing, _ = self._generate(kind, flags=(None, None))
                self.assertEqual([self._key(c) for c in same], [self._key(c) for c in basic])
                self.assertEqual([self._key(c) for c in missing], [self._key(c) for c in basic])
                with mock.patch.dict(os.environ, {
                    "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
                    "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
                }):
                    expanded, _ = self._generate(kind, flags=(True, False))
                self.assertEqual([self._key(c) for c in basic], [self._key(c) for c in expanded])

    def test_pointwise_switch_preserves_candidates_and_group_planning(self):
        for grouped in (False, True):
            for flags in ((False, False), (False, True), (True, False), (True, True)):
                with self.subTest(grouped=grouped, flags=flags):
                    on, on_meta = self._generate(flags=flags, grouped=grouped)
                    configs, meta = self._generate(
                        flags=flags, grouped=grouped, autotune_pointwise=False,
                    )
                    self.assertEqual([self._key(c) for c in on], [self._key(c) for c in configs])
                    if grouped:
                        self.assertEqual(on_meta["grouped_candidate_plan"], meta["grouped_candidate_plan"])
                        counts = [len(meta["grouped_candidate_plan"]["group_to_candidates"][g])
                                  for g in meta["grouped_candidate_plan"]["reachable_group_ids"]]
                    else:
                        counts = [len(configs)]
                    self.assertTrue(all(count > 1 for count in counts))

    def test_pointwise_switch_does_not_change_real_reduction_sets(self):
        for kind in ("reduction", "persistent"):
            for grouped in (False, True):
                for flags in ((False, False), (False, True), (True, False), (True, True)):
                    with self.subTest(kind=kind, grouped=grouped, flags=flags):
                        on, on_meta = self._generate(kind, flags, grouped=grouped)
                        off, off_meta = self._generate(
                            kind, flags, grouped=grouped, autotune_pointwise=False,
                        )
                        self.assertEqual([self._key(c) for c in on], [self._key(c) for c in off])
                        if grouped:
                            self.assertEqual(on_meta["grouped_candidate_plan"], off_meta["grouped_candidate_plan"])

    def test_grouped_max_preserves_each_group_and_plan_consistency(self):
        for kind in ("pointwise", "reduction", "persistent"):
            with self.subTest(kind=kind):
                _, basic_meta = self._generate(kind, grouped=True)
                basic = basic_meta["grouped_candidate_plan"]
                for flags in ((False, True), (True, False), (True, True)):
                    configs, meta = self._generate(kind, flags, grouped=True)
                    plan = meta["grouped_candidate_plan"]
                    self.assertEqual(plan, basic)
                    self.assertEqual(len(configs), len(plan["variants"]))
                    self.assertEqual(plan["reachable_group_ids"], basic["reachable_group_ids"])
                    for group in plan["reachable_group_ids"]:
                        self.assertGreater(len(plan["group_to_candidates"][group]), 1)
                        for candidate in plan["group_to_candidates"][group]:
                            self.assertIn(candidate["variant_id"], plan["variants"])
                            self.assertIn(candidate["policy_id"], plan["policies"])
                            self.assertEqual(candidate["selection_key"], group)
                    self.assertEqual(sum(map(len, plan["group_to_candidates"])),
                                     len(plan["candidate_entries"]))

    def test_fasta_is_independent_of_max_and_preserves_selection_paths(self):
        with mock.patch.object(npu_config, "fasta_autotune", True):
            for flags in ((False, False), (False, True), (True, False), (True, True)):
                meta = {"max_autotune": flags[0], "max_autotune_pointwise": flags[1]}
                for is_reduction in (False, True):
                    self.assertTrue(triton_heuristics._use_fasta_autotune(meta, is_reduction))
                    self.assertFalse(triton_heuristics._use_fasta_autotune(
                        dict(meta, group_enabled=True), is_reduction,
                    ))
                    for flag in ("deterministic", "are_deterministic_algorithms_enabled"):
                        self.assertEqual(triton_heuristics._use_fasta_autotune(
                            dict(meta, **{flag: True}), is_reduction,
                        ), not is_reduction)
        with mock.patch.object(npu_config, "fasta_autotune", False):
            self.assertFalse(triton_heuristics._use_fasta_autotune({"max_autotune": True}))

    def test_fasta_generator_is_used_without_max_flags(self):
        generator = mock.Mock()
        expected, _ = self._generate()
        generator.descend_split_tiling.return_value = expected
        factory = mock.Mock(return_value=generator)
        module = SimpleNamespace(FastATileGenerator=factory)
        with (
            mock.patch.object(npu_config, "fasta_autotune", True),
            mock.patch.dict(sys.modules, {
                "torch_npu._inductor.runtime.fasta_autotune": module,
            }),
        ):
            configs, _ = self._generate()
        factory.assert_called_once()
        generator.descend_split_tiling.assert_called_once_with()
        self.assertEqual([self._key(c) for c in configs], [self._key(c) for c in expected])

    def test_explicit_autotuner_takes_precedence_over_fasta_and_grouping(self):
        fast = mock.Mock()
        module = SimpleNamespace(NPUFastAutotuner=fast)
        with (
            mock.patch.object(npu_config, "fasta_autotune", True),
            mock.patch.dict(sys.modules, {
                "torch_npu._inductor.runtime.fasta_autotune": module,
            }),
            mock.patch.object(triton_heuristics, "NPUCachingAutotuner") as basic,
            mock.patch.object(triton_heuristics, "NPUSymbolicGroupedAutotuner") as grouped,
        ):
            for heuristic in (HeuristicType.POINTWISE, HeuristicType.REDUCTION,
                              HeuristicType.PERSISTENT_REDUCTION):
                for group_enabled in (False, True):
                    for deterministic in (False, True):
                        with self.subTest(heuristic=heuristic, grouped=group_enabled,
                                          deterministic=deterministic):
                            explicit = mock.Mock()
                            result = triton_heuristics.cached_autotune(
                                None, [Config({})], {}, heuristic,
                                inductor_meta={
                                    "force_disable_caches": True,
                                    "group_enabled": group_enabled,
                                    "deterministic": deterministic,
                                },
                                caching_autotuner_cls=explicit,
                            )(object())
                            self.assertIs(result, explicit.return_value)
                            explicit.assert_called_once()
                            fast.assert_not_called()
                            basic.assert_not_called()
                            grouped.assert_not_called()

    def test_cached_autotune_dispatch_uses_the_same_fasta_policy(self):
        fast = mock.Mock()
        module = SimpleNamespace(NPUFastAutotuner=fast)
        with (
            mock.patch.object(npu_config, "fasta_autotune", True),
            mock.patch.dict(sys.modules, {
                "torch_npu._inductor.runtime.fasta_autotune": module,
            }),
            mock.patch.object(triton_heuristics, "NPUCachingAutotuner") as basic,
            mock.patch.object(triton_heuristics, "NPUSymbolicGroupedAutotuner") as grouped,
        ):
            for heuristic in (HeuristicType.POINTWISE, HeuristicType.REDUCTION,
                              HeuristicType.PERSISTENT_REDUCTION):
                for flags in ((False, False), (False, True), (True, False), (True, True)):
                    for group_enabled in (False, True):
                        for deterministic_field in (None, "deterministic",
                                                    "are_deterministic_algorithms_enabled"):
                            fast.reset_mock()
                            basic.reset_mock()
                            grouped.reset_mock()
                            meta = {
                                "max_autotune": flags[0], "max_autotune_pointwise": flags[1],
                                "force_disable_caches": True, "group_enabled": group_enabled,
                            }
                            if deterministic_field:
                                meta[deterministic_field] = True
                            with self.subTest(heuristic=heuristic, flags=flags,
                                              grouped=group_enabled, deterministic=deterministic_field):
                                result = triton_heuristics.cached_autotune(
                                    None, [Config({})], {}, heuristic, inductor_meta=meta,
                                )(object())
                                if group_enabled:
                                    expected = grouped
                                elif not (
                                    deterministic_field and heuristic != HeuristicType.POINTWISE
                                ):
                                    expected = fast
                                else:
                                    expected = basic
                                self.assertIs(result, expected.return_value)
                                expected.assert_called_once()
                                for unselected in (fast, basic, grouped):
                                    if unselected is not expected:
                                        unselected.assert_not_called()


if __name__ == "__main__":
    run_tests()
