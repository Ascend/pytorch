# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Deterministic reduction selection and real compiled NPU numerical replay."""

from types import SimpleNamespace
from unittest import mock

import torch
import torch_npu  # noqa: F401
from torch.testing._internal.common_utils import (
    TestCase, instantiate_parametrized_tests, parametrize, run_tests,
)
import torch_npu._inductor.triton_experimental.npu_triton_heuristics as _heur


def _norm_list(configs):
    return [_heur._npu_config_key(config) for config in configs]


class TestDeterministicAutotune(TestCase):
    def _candidates(self, configs, kind, metadata):
        with mock.patch.object(_heur.AutotuneCache, "create", return_value=None) as cache:
            with mock.patch.object(_heur, "NPUCachingAutotuner") as autotuner:
                decorator = _heur.cached_autotune(
                    {"x": 768, "r0_": 128}, configs, {}, kind,
                    filename=__file__, inductor_meta=dict(metadata),
                )
                decorator(SimpleNamespace(fn=lambda XBLOCK: None))
                return autotuner.call_args.kwargs["configs"], cache.call_count

    @parametrize("coordesc", [False, True])
    @parametrize("kind", [_heur.HeuristicType.REDUCTION, _heur.HeuristicType.PERSISTENT_REDUCTION])
    @parametrize("flag", ["deterministic", "are_deterministic_algorithms_enabled", "force_filter_reduction_configs"])
    def test_reduction_filters_before_timing_cache(self, kind, flag, coordesc):
        configs = [_heur.Config({"XBLOCK": 64, "R0_BLOCK": rb}, num_warps=4) for rb in (32, 64, 128)]
        first, cache_calls = self._candidates(configs, kind, {flag: True, "coordinate_descent_tuning": coordesc})
        second, _ = self._candidates(configs, kind, {flag: True, "coordinate_descent_tuning": coordesc})
        self.assertEqual(len(first), 1)
        self.assertEqual(_norm_list(first), _norm_list(second))
        self.assertEqual(first[0].kwargs, {"XBLOCK": 64, "R0_BLOCK": 64})
        self.assertEqual(cache_calls, 0)

    def test_nondeterministic_reduction_preserves_backend_options(self):
        configs = [_heur.npu_triton_config({"x": 768}, 64, auto_blockify_size=x) for x in (2, 4, 8)]
        actual, _ = self._candidates(configs, _heur.HeuristicType.REDUCTION,
                                     {"are_deterministic_algorithms_enabled": False})
        self.assertEqual(_norm_list(actual), _norm_list(configs))
        self.assertEqual(len(actual), 3)

    def test_deterministic_pointwise_still_autotunes(self):
        configs = [_heur.Config({"XBLOCK": x}) for x in (64, 128)]
        actual, cache_calls = self._candidates(configs, _heur.HeuristicType.POINTWISE,
                                               {"are_deterministic_algorithms_enabled": True})
        self.assertEqual(_norm_list(actual), _norm_list(configs))
        self.assertEqual(cache_calls, 1)


instantiate_parametrized_tests(TestDeterministicAutotune)


class TestDeterministicReductionNumerics(TestCase):
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    @parametrize("noncontiguous", [False, True])
    def test_fresh_compilation_forward_and_backward(self, dtype, noncontiguous):
        if not torch.npu.is_available():
            self.skipTest("requires an NPU")

        from torch._inductor import config as inductor_config

        generator = torch.Generator().manual_seed(42)
        inputs = [torch.randn(1, 192, 4, 128, generator=generator, dtype=dtype).to("npu") for _ in range(2)]
        if noncontiguous:
            inputs = [x.transpose(1, 2) for x in inputs]

        def reduction(x, y):
            result = (x.float() * y.float()).sum(dim=-1)
            return result, result.square().sum()

        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        results = []
        try:
            torch.use_deterministic_algorithms(True)
            for _ in range(2):
                torch._dynamo.reset()
                args = [x.detach().clone(memory_format=torch.preserve_format).requires_grad_() for x in inputs]
                with inductor_config.patch(force_disable_caches=True, max_autotune=False):
                    compiled = torch.compile(reduction, fullgraph=True, options={"npu_backend": "triton_experimental"})
                    output, loss = compiled(*args)
                    gradients = torch.autograd.grad(loss, args)
                results.append([x.detach().cpu() for x in (output, loss, *gradients)])
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            torch._dynamo.reset()
        for first, second in zip(*results):
            self.assertEqual(first, second, rtol=0, atol=0)


instantiate_parametrized_tests(TestDeterministicReductionNumerics)


if __name__ == "__main__":
    run_tests()
