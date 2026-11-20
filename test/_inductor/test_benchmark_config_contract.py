import contextlib
import copy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import triton
from torch._inductor.runtime.triton_heuristics import config_to_dict
from torch._inductor import wrapper_benchmark
from torch._inductor.codecache import PyCodeCache

from torch_npu._inductor.runtime.triton_heuristics import NPUCachingAutotuner


class TestBenchmarkConfigContract(unittest.TestCase):
    def make_tuner(self, blocks=(32, 64)):
        launcher = SimpleNamespace(n_regs=12, n_spills=0, shared=256,
                                   config=triton.Config({"compile_mode": "simt"}))
        candidates = [dict(candidate_id=f"c{i}", variant_id="v0",
                           full_config=config_to_dict(triton.Config(
                               {"XBLOCK": block, "split_blocks": [block]}, num_warps=1)))
                      for i, block in enumerate(blocks)]
        timings = {c["candidate_id"]: 0.25 + i for i, c in enumerate(candidates)}
        tuner = object.__new__(NPUCachingAutotuner)
        tuner.candidate_plan = {"candidate_entries": candidates}
        tuner.compiled_candidate_entries = tuple(
            dict(candidate=c, launcher=launcher) for c in candidates)
        tuner._benchmark_candidate_entries = lambda *a, **kw: timings
        return tuner, launcher, timings

    def test_shared_launcher_keeps_all_full_configs(self):
        tuner, launcher, timings = self.make_tuner()
        before = copy.deepcopy(tuner.candidate_plan)
        result = tuner.benchmark_all_configs()
        self.assertEqual(len(result), 2)
        self.assertEqual([k.config.kwargs["XBLOCK"] for k in result], [32, 64])
        self.assertEqual(list(result.values()), list(timings.values()))
        self.assertTrue(all((k.n_regs, k.n_spills, k.shared) == (12, 0, 256) for k in result))
        self.assertEqual(tuner.candidate_plan, before)
        self.assertEqual(launcher.config.kwargs, {"compile_mode": "simt"})
        self.assertIs(tuner._benchmark_candidate_entries(), timings)

    def test_equal_configs_still_have_distinct_candidate_results(self):
        tuner, _, _ = self.make_tuner((32, 32))
        self.assertEqual(len(tuner.benchmark_all_configs()), 2)

    def test_uncompiled_candidates_are_not_reported_as_launchers(self):
        tuner, _, timings = self.make_tuner()
        timings["uncompiled"] = float("inf")
        self.assertEqual(len(tuner.benchmark_all_configs()), 2)

    def test_errors_are_not_hidden(self):
        tuner, _, _ = self.make_tuner()
        def fail(*args, **kwargs):
            raise RuntimeError("all candidates failed")
        tuner._benchmark_candidate_entries = fail
        with self.assertRaisesRegex(RuntimeError, "all candidates failed"):
            tuner.benchmark_all_configs()

    def test_torch_wrapper_can_print_every_candidate(self):
        tuner, _, _ = self.make_tuner()
        kernel = SimpleNamespace(device_props=SimpleNamespace(type="npu"),
                                 fn=SimpleNamespace(arg_names=[]), inductor_meta={"kernel_num_gb": 1.0})
        module = SimpleNamespace(key="contract", call=lambda args: None, get_args=lambda: (),
                                 benchmark_all_configs=lambda args: tuner.benchmark_all_configs())
        output = io.StringIO()
        with patch.object(wrapper_benchmark, "get_triton_kernel", return_value=kernel), \
             patch.object(wrapper_benchmark, "get_kernel_category", return_value="pointwise"), \
             patch.object(PyCodeCache, "modules", [module]), \
             contextlib.redirect_stdout(output):
            module.__file__ = "/tmp/test_benchmark_contract.py"
            wrapper_benchmark.benchmark_all_kernels("contract", True)
        self.assertEqual(output.getvalue().count(" regs "), 2)
        self.assertIn("XBLOCK: 32", output.getvalue())
        self.assertIn("XBLOCK: 64", output.getvalue())


if __name__ == "__main__":
    unittest.main()
