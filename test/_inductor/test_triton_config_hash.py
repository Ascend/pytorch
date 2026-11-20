import copy

import triton
from torch._inductor.runtime import (
    coordinate_descent_tuner,
    runtime_utils,
    triton_heuristics,
)
from torch.testing._internal.common_utils import run_tests, TestCase

import torch_npu  # noqa: F401
from torch_npu._inductor.runtime.runtime_utils import (
    _original_triton_config_to_hashable,
    patch_triton_config_to_hashable,
    triton_config_to_hashable,
)


class TestTritonConfigHash(TestCase):
    def test_existing_keys_are_preserved(self):
        cfg = triton.Config({"XBLOCK": 128, "axes": (0, 1), "enabled": True})
        expected = _original_triton_config_to_hashable(cfg)
        actual = triton_config_to_hashable(cfg)
        self.assertEqual(actual, expected)
        self.assertEqual(hash(actual), hash(expected))

    def test_nested_sequences_are_frozen_without_mutating_config(self):
        cfg = triton.Config({"axes": [0, [1, 2], (3, [4])]})
        original = copy.deepcopy(cfg.kwargs)
        key = triton_config_to_hashable(cfg)
        equivalent = triton.Config({"axes": (0, (1, 2), (3, (4,)))})
        self.assertEqual(key, triton_config_to_hashable(equivalent))
        self.assertEqual({key: 1.25}[triton_config_to_hashable(equivalent)], 1.25)
        self.assertEqual(cfg.kwargs, original)
        self.assertIsInstance(cfg.kwargs["axes"], list)
        cfg.kwargs["axes"][1].append(9)
        self.assertNotEqual(key, triton_config_to_hashable(cfg))

    def test_distinct_configs_remain_distinct(self):
        configs = [
            triton.Config({"axes": [0, 1]}, num_warps=4, num_stages=2),
            triton.Config({"axes": [1, 0]}, num_warps=4, num_stages=2),
            triton.Config({"axes": [0, 2]}, num_warps=4, num_stages=2),
            triton.Config({"axes": [0, 1]}, num_warps=8, num_stages=2),
            triton.Config({"axes": [0, 1]}, num_warps=4, num_stages=3),
        ]
        keys = {triton_config_to_hashable(cfg) for cfg in configs}
        self.assertEqual(len(keys), len(configs))

    def test_patch_covers_cache_and_deduplication(self):
        patch_triton_config_to_hashable()
        patch_triton_config_to_hashable()
        for module in (runtime_utils, coordinate_descent_tuner, triton_heuristics):
            self.assertIs(module.triton_config_to_hashable, triton_config_to_hashable)

        cfg = triton.Config({"axes": [0, [1, 2]]})
        equivalent = triton.Config({"axes": (0, (1, 2))})
        tuner = coordinate_descent_tuner.CoordescTuner()
        tuner.cache_benchmark_result(cfg, 1.25)
        self.assertEqual(tuner.lookup_in_cache(equivalent), 1.25)
        self.assertEqual(len(triton_heuristics.unique_configs([cfg, equivalent])), 1)


if __name__ == "__main__":
    run_tests()
