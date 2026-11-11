from types import SimpleNamespace
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase
from torch._inductor.runtime.triton_heuristics import Config, HeuristicType

from torch_npu._inductor.codegen.tile_generator import ub_overflow_retry_configs
from torch_npu._inductor.runtime.triton_heuristics import (
    NPUCachingAutotuner, build_candidate_plan,
)


class TestUBOverflowRetry(TestCase):
    def setUp(self):
        super().setUp()
        import contextlib
        patcher = mock.patch("torch._dynamo.device_interface.DeviceGuard", return_value=contextlib.nullcontext())
        patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self, size):
        return Config({"XBLOCK": size, "compile_mode": "simd_simt_template"}, num_warps=1)

    def _meta(self):
        return {"axis_names": ["x"], "tiling_axis": [0], "inductor_ascend_linear_mode": "no_linear"}

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
        tuner.get_device_interface = lambda: SimpleNamespace(synchronize=lambda _: None, current_device=lambda: 0)
        tuner.get_fn_name = lambda: "ub_retry_test"
        return tuner

    @staticmethod
    def _ub_error(size):
        return RuntimeError(f"ub overflow, requires {size * 512} bits while 1769472 bits available!")

    def test_block_names_follow_v210_codegen_modes(self):
        for mode, name in (("linear", "XBLOCK_SUB"), ("no_linear", "XBLOCK"),
                           ("no_linear_loop", "XBLOCK_SUB")):
            with self.subTest(mode=mode):
                cfg = Config({name: 4096})
                meta = {**self._meta(), "inductor_ascend_linear_mode": mode}
                extra = ub_overflow_retry_configs([cfg], [(cfg, 2097152, 1769472)], meta, {name})
                self.assertEqual([candidate.kwargs[name] for candidate in extra], [2048, 1024, 512, 256])
                self.assertEqual(cfg.kwargs[name], 4096)

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


if __name__ == "__main__":
    run_tests()
