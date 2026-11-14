"""Arithmetic checks for combo split boundaries; NPU accuracy is tested separately."""
import copy
import unittest
from unittest.mock import patch

import torch

from torch_npu._inductor import config
from torch_npu._inductor.runtime.combo_tile_generator import ComboTileGenerator
from torch_npu._inductor.runtime.tile_generator import aligned_numel_32byte


def subkernel(extent, inner=128, persistent=False):
    return {
        'axis_names': ['x0', 'r0' if persistent else 'x1'],
        'size_hints': {'x0': extent, 'r0' if persistent else 'x1': inner},
        'split_axis': [0],
        'tiling_axis': [0, 1],
        'no_loop_axis': [],
        'low_dims': [1],
        'split_axis_dtype': torch.float32,
        'dual_reduction': False,
        'persistent_reduction': persistent,
    }


class TestComboTileGenerator(unittest.TestCase):
    def setUp(self):
        self.core_count_patch = patch.object(config, '_get_core_nums', return_value=(None, 1, 56))
        self.core_count_patch.start()
        self.addCleanup(self.core_count_patch.stop)

    def test_candidates_fit_wave_budget_and_ignore_subkernel_order(self):
        kernels = [subkernel(x) for x in (555, 423, 111)]
        original = copy.deepcopy(kernels)
        first = ComboTileGenerator(kernels)
        candidates = first.descend_split_tiling()
        self.assertTrue(candidates)
        for candidate in candidates:
            block = candidate.kwargs['X0BLOCK']
            self.assertLessEqual(first.calculate_total_programs([block, 128]),
                                 config.num_vector_core * config.combo_kernel_max_waves)
            self.assertIn('X1BLOCK_SUB', candidate.kwargs)
        second = ComboTileGenerator(list(reversed(kernels)))
        other = second.descend_split_tiling()

        def normalize(values):
            return {tuple(sorted(item.kwargs.items())) for item in values}

        self.assertEqual(normalize(candidates), normalize(other))
        self.assertEqual(kernels, original)

    def test_search_reaches_each_wave_boundary_and_keeps_one_wave_configs(self):
        cores = 8
        kernels = [subkernel(794624, 1), subkernel(3178496, 1)]
        with patch.object(config, "_get_core_nums", return_value=(None, 1, cores)):
            with patch.object(config, "combo_kernel_max_waves", 1):
                one_wave = ComboTileGenerator(copy.deepcopy(kernels)).descend_split_tiling()
            generator = ComboTileGenerator(copy.deepcopy(kernels))
            candidates = generator.descend_split_tiling()

            def key(candidate):
                return (
                    tuple(sorted(candidate.kwargs.items())),
                    candidate.num_warps,
                    candidate.num_stages,
                )

            self.assertLessEqual({key(candidate) for candidate in one_wave},
                                 {key(candidate) for candidate in candidates})
            programs = [
                generator.calculate_total_programs([candidate.kwargs["X0BLOCK"], 1])
                for candidate in candidates
            ]
            self.assertTrue(any(programs_per_wave > cores for programs_per_wave in programs))
            self.assertLessEqual(max(programs), cores * generator.max_waves)
            for waves in range(1, generator.max_waves + 1):
                limit = cores * waves
                with patch.object(generator, "_program_limit", limit):
                    boundary = generator.calcu_last_split_blocks(0)
                aligned_boundary = aligned_numel_32byte(boundary, generator.dtype_bytes)
                self.assertTrue(any(candidate.kwargs["X0BLOCK"] == aligned_boundary for candidate in candidates))
                self.assertLessEqual(generator.calculate_total_programs([aligned_boundary, 1]), limit)
                self.assertLessEqual(generator.calculate_total_programs([boundary, 1]), limit)
                self.assertGreater(generator.calculate_total_programs([boundary - 1, 1]), limit)
            self.assertEqual(generator.max_programs, cores * config.combo_kernel_max_waves)

    def test_full_subblock_is_kept_for_each_retained_split(self):
        kernels = [subkernel(x, 1) for x in (49664, 198656, 297984)]
        for item in kernels:
            item['split_axis_dtype'] = torch.bfloat16
        generator = ComboTileGenerator(kernels)
        candidates = generator.descend_split_tiling()
        minimum = generator.calcu_last_split_blocks(0)
        self.assertTrue(any(
            item.kwargs['X0BLOCK'] > minimum
            and item.kwargs['X0BLOCK_SUB'] >= item.kwargs['X0BLOCK']
            for item in candidates
        ))

    def test_persistent_extent_uses_actual_branch_shapes(self):
        generator = ComboTileGenerator([subkernel(100, 32, True),
                                       subkernel(200, 512, True)])
        self.assertEqual(generator.calculate_config_numel(
            {'X0BLOCK_SUB': 8, 'R0BLOCK_SUB': 1}), 8 * 512)

    def test_mixed_dtype_footprint_uses_common_dtype_units(self):
        kernels = [subkernel(208, 788, True), subkernel(388, 20, True)]
        kernels[1]['split_axis_dtype'] = torch.int64
        generator = ComboTileGenerator(kernels, input_ptr_num=3)
        self.assertEqual(
            generator.calculate_config_numel({'X0BLOCK_SUB': 24}),
            24 * 788 // 2,
        )

    def test_no_split_subkernel_still_has_one_program(self):
        item = subkernel(1)
        item['split_axis'] = []
        generator = ComboTileGenerator([item, copy.deepcopy(item)])
        self.assertEqual(generator.calculate_total_programs([1, 128]), 2)
        self.assertTrue(generator.descend_split_tiling())


if __name__ == '__main__':
    unittest.main()
