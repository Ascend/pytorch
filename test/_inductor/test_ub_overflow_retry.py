from types import SimpleNamespace
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.runtime.triton_heuristics import Config, HeuristicType, NoTritonConfigsError

from torch_npu._inductor.runtime.tile_generator import ub_overflow_retry_configs
from torch_npu._inductor.runtime.triton_heuristics import (
    NPUCachingAutotuner, NPUCostModelAutotuner, build_candidate_plan,
)


class TestUBOverflowRetry(TestCase):
    def _config(self, size):
        return Config({"XBLOCK": size, "compile_mode": "simd_simt_template"}, num_warps=1)

    def _meta(self):
        return {"axis_names": ["x"], "tiling_axis": [0], "requires_no_linear_block_remap": True}

    def _tuner(self, configs, cls=NPUCachingAutotuner):
        tuner = object.__new__(cls)
        tuner.configs = configs
        tuner.triton_meta = {"device": 0}
        tuner.inductor_meta = self._meta()
        tuner.runtime_block_arg_names = ()
        tuner.candidate_plan = build_candidate_plan(configs, ())
        tuner.compile_results = []
        tuner.launchers = []
        tuner.custom_kernel = False
        tuner.heuristic_type = HeuristicType.POINTWISE
        tuner.fn = SimpleNamespace(arg_names=["XBLOCK"], constexprs=[0])
        tuner.get_fn_name = lambda: "ub_retry_test"
        return tuner

    @staticmethod
    def _ub_error(size):
        return RuntimeError(f"ub overflow, requires {size * 512} bits while 1769472 bits available!")

    def test_feedback_bounded_below_smallest_tile(self):
        configs = [self._config(size) for size in (32768, 16384, 8192, 4096)]
        failures = [(cfg, cfg.kwargs["XBLOCK"] * 512, 1769472) for cfg in configs]
        extra = ub_overflow_retry_configs(configs, failures, self._meta(), {"XBLOCK"})
        self.assertEqual([cfg.kwargs["XBLOCK"] for cfg in extra], [2048, 1024, 512, 256])
        self.assertEqual([cfg.kwargs["XBLOCK"] for cfg in configs], [32768, 16384, 8192, 4096])
        cfg = self._config(2)
        extra = ub_overflow_retry_configs([cfg], [(cfg, 100000, 1)], self._meta(), {"XBLOCK"})
        self.assertEqual([candidate.kwargs["XBLOCK"] for candidate in extra], [1])

    def test_fixed_and_runtime_axes_are_preserved(self):
        cfg = Config({"X0BLOCK": 64, "X0BLOCK_SUB": 32, "R1BLOCK_SUB": 512,
                      "split_blocks": (64,), "compile_mode": "simd_simt_template"})
        compiled_cfg = Config({name: value for name, value in cfg.kwargs.items()
                               if name not in ("X0BLOCK", "split_blocks")})
        meta = {"axis_names": ["x0", "r1"], "tiling_axis": [0, 1]}
        failures = [(compiled_cfg, 41984000, 2031616)]
        constexprs = {"X0BLOCK_SUB", "R1BLOCK_SUB"}
        extra = ub_overflow_retry_configs(
            [cfg], failures, meta, constexprs, ("X0BLOCK",), persistent_reduction=True,
        )
        self.assertTrue(extra)
        for candidate in extra:
            self.assertEqual(candidate.kwargs["X0BLOCK"], 64)
            self.assertEqual(candidate.kwargs["R1BLOCK_SUB"], 512)
            self.assertEqual(candidate.kwargs["split_blocks"], (64,))
            self.assertLess(candidate.kwargs["X0BLOCK_SUB"], 32)
        for metadata, names in (({**meta, "no_loop_axis": [0]}, constexprs), (meta, set())):
            self.assertEqual(ub_overflow_retry_configs(
                [cfg], failures, metadata, names, ("X0BLOCK",), persistent_reduction=True,
            ), [])

    def test_retry_waits_for_batch_and_publishes_success(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                tuner = self._tuner([self._config(8192), self._config(4096)])
                original_configs, original_plan = tuner.configs, tuner.candidate_plan
                calls = []

                def compile_config(cfg):
                    self.assertIs(tuner.configs, original_configs)
                    self.assertIs(tuner.candidate_plan, original_plan)
                    size = cfg.kwargs["XBLOCK"]
                    calls.append(size)
                    if size >= 4096:
                        raise self._ub_error(size)
                    return SimpleNamespace(config=cfg)

                search = tuner._retry_with_smaller_configs

                def search_after_batch(compile_fn, failures):
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(len(failures), 2)
                    return search(compile_fn, failures)

                tuner._precompile_config = compile_config
                compile_fn = tuner._precompile_configs_parallel if parallel else tuner._precompile_configs
                with mock.patch.object(tuner, "_retry_with_smaller_configs", side_effect=search_after_batch):
                    tuner._precompile_worker_with(compile_fn)
                self.assertEqual(sorted(calls), [256, 512, 1024, 2048, 4096, 8192])
                self.assertIsNot(tuner.configs, original_configs)
                self.assertEqual(len(original_configs), 2)
                self.assertEqual(len(tuner.configs), 6)
                self.assertEqual(len(tuner.candidate_plan["candidate_entries"]), 6)
                self.assertEqual({result.config._runtime_variant_id for result in tuner.compile_results},
                                 {"v2", "v3", "v4", "v5"})

    def test_only_all_ub_failure_retries(self):
        for parallel in (False, True):
            for scenario in ("success", "non_ub_first", "non_ub_last", "invalid_capacity"):
                with self.subTest(parallel=parallel, scenario=scenario):
                    tuner = self._tuner([self._config(8192), self._config(4096)])
                    original_configs, original_plan = tuner.configs, tuner.candidate_plan
                    calls = []

                    def compile_config(cfg):
                        size = cfg.kwargs["XBLOCK"]
                        calls.append(size)
                        if scenario == "success" and size == 4096:
                            return SimpleNamespace(config=cfg)
                        if (scenario == "non_ub_first" and size == 8192
                                or scenario == "non_ub_last" and size == 4096):
                            raise RuntimeError("unrelated lowering error")
                        if scenario == "invalid_capacity":
                            raise RuntimeError("ub overflow, requires 1024 bits while 2048 bits available!")
                        raise self._ub_error(size)

                    tuner._precompile_config = compile_config
                    compile_fn = tuner._precompile_configs_parallel if parallel else tuner._precompile_configs
                    with mock.patch.object(tuner, "_retry_with_smaller_configs") as retry:
                        if scenario == "success":
                            tuner._precompile_worker_with(compile_fn)
                            self.assertEqual(len(tuner.compile_results), 1)
                        else:
                            with self.assertRaises(NoTritonConfigsError):
                                tuner._precompile_worker_with(compile_fn)
                        retry.assert_not_called()
                    self.assertEqual(sorted(calls), [4096, 8192])
                    self.assertIs(tuner.configs, original_configs)
                    self.assertIs(tuner.candidate_plan, original_plan)

    def test_failed_retry_preserves_error_and_is_not_repeated(self):
        for parallel in (False, True):
            for extra_ub in (False, True):
                with self.subTest(parallel=parallel, extra_ub=extra_ub):
                    tuner = self._tuner([self._config(4096)])
                    original_configs, original_plan = tuner.configs, tuner.candidate_plan
                    calls, errors = [], []

                    def compile_config(cfg):
                        size = cfg.kwargs["XBLOCK"]
                        calls.append(size)
                        if size < 4096 and not extra_ub:
                            raise RuntimeError("retry lowering error")
                        raise self._ub_error(size)

                    tuner._precompile_config = compile_config
                    compile_batch = tuner._precompile_configs_parallel if parallel else tuner._precompile_configs

                    def compile_fn(configs):
                        try:
                            return compile_batch(configs)
                        except NoTritonConfigsError as error:
                            errors.append(error)
                            raise

                    with self.assertRaises(NoTritonConfigsError) as caught:
                        tuner._precompile_worker_with(compile_fn)
                    self.assertEqual(len(errors), 2)
                    self.assertIs(caught.exception, errors[0])
                    self.assertIsNone(caught.exception.__context__)
                    self.assertIn("requires 2097152 bits", str(caught.exception))
                    self.assertEqual(sorted(calls), [256, 512, 1024, 2048, 4096])
                    self.assertTrue(tuner._ub_retry_attempted)
                    self.assertIs(tuner.configs, original_configs)
                    self.assertIs(tuner.candidate_plan, original_plan)
                    self.assertEqual(tuner.compile_results, [])
                    with self.assertRaises(NoTritonConfigsError):
                        tuner._precompile_worker_with(compile_fn)
                    self.assertEqual(len(calls), 6)

    def test_custom_and_grouped_kernels_do_not_expand(self):
        for custom, grouped in ((True, False), (False, True)):
            with self.subTest(custom=custom, grouped=grouped):
                tuner = self._tuner([self._config(4096)])
                tuner.custom_kernel = custom
                tuner.inductor_meta["group_enabled"] = grouped
                tuner._precompile_config = mock.Mock(side_effect=self._ub_error(4096))
                with self.assertRaises(NoTritonConfigsError):
                    tuner._precompile_worker_with(tuner._precompile_configs)
                tuner._precompile_config.assert_called_once()
                self.assertEqual(len(tuner.configs), 1)
                self.assertFalse(getattr(tuner, "_ub_retry_attempted", False))

    def test_costmodel_keeps_existing_fallback_without_ub_retry(self):
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                tuner = self._tuner([self._config(8192)], NPUCostModelAutotuner)
                tuner._costmodel_fallback_configs = [self._config(2048)]
                calls = []

                def compile_config(cfg):
                    size = cfg.kwargs["XBLOCK"]
                    calls.append(size)
                    if size >= 4096:
                        raise self._ub_error(size)
                    return SimpleNamespace(config=cfg)

                tuner._precompile_config = compile_config
                compile_fn = tuner._precompile_configs_parallel if parallel else tuner._precompile_configs
                with mock.patch.object(tuner, "_retry_with_smaller_configs") as retry:
                    tuner._precompile_worker_with(compile_fn)
                    retry.assert_not_called()
                self.assertEqual(calls, [8192, 2048])
                self.assertFalse(getattr(tuner, "_ub_retry_attempted", False))


if __name__ == "__main__":
    run_tests()
