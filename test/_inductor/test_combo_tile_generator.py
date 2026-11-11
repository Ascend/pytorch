"""Arithmetic checks for combo split boundaries; NPU accuracy is tested separately."""
import copy
import unittest
from unittest.mock import patch

import torch

from torch_npu._inductor import config
from torch_npu._inductor.codegen.combo_tile_generator import ComboTileGenerator
from torch_npu._inductor.codegen.tile_generator import TileGenerator


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
    def test_boundary_matches_exhaustive_search(self):
        for extents in ([555, 423, 111], [101] * 8, [1, 1, 1003], [1] * config.num_vector_core):
            with self.subTest(extents=extents):
                generator = ComboTileGenerator([subkernel(x) for x in extents])
                actual = generator.calcu_last_split_blocks(0)
                expected = next(
                    block for block in range(1, max(extents) + 1)
                    if sum((x + block - 1) // block for x in extents)
                    <= config.num_vector_core * config.combo_kernel_max_waves
                )
                self.assertEqual(actual, expected)

    def test_tail_correction_is_required(self):
        limit = config.num_vector_core * config.combo_kernel_max_waves
        extents = [limit + 1] * (limit - 1)
        generator = ComboTileGenerator([subkernel(x) for x in extents])
        naive = (sum(extents) + limit - 1) // limit
        self.assertGreater(generator.calculate_total_programs([naive, 128]), limit)
        self.assertLessEqual(generator.calculate_total_programs(
            [generator.calcu_last_split_blocks(0), 128]), limit)

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
        normalize = lambda values: {tuple(sorted(item.kwargs.items())) for item in values}
        self.assertEqual(normalize(candidates), normalize(other))
        self.assertEqual(kernels, original)

    def test_search_reaches_each_wave_boundary(self):
        # Keep the eight-wave boundary above the existing small-tile stop threshold.
        generator = ComboTileGenerator([subkernel(794624, 1), subkernel(3178496, 1)])
        candidates = generator.descend_split_tiling()
        programs = [generator.calculate_total_programs([c.kwargs['X0BLOCK'], 1])
                    for c in candidates]
        self.assertTrue(any(n > config.num_vector_core for n in programs))
        self.assertLessEqual(max(programs), config.num_vector_core * config.combo_kernel_max_waves)
        for waves in range(1, generator.max_waves + 1):
            limit = config.num_vector_core * waves
            with patch.object(generator, '_program_limit', limit):
                boundary = generator.calcu_last_split_blocks(0)
            self.assertTrue(any(c.kwargs['X0BLOCK'] == boundary for c in candidates))
            self.assertLessEqual(generator.calculate_total_programs([boundary, 1]), limit)
            self.assertGreater(generator.calculate_total_programs([boundary - 1, 1]), limit)

    def test_complete_one_wave_configs_are_retained(self):
        for kernels in ([subkernel(373, 16384)],
                        [subkernel(7833, 98), subkernel(22753, 208)],
                        [subkernel(10, 256, True), subkernel(19, 256, True)]):
            reference = ComboTileGenerator(copy.deepcopy(kernels))
            with patch.object(reference, '_program_limit', config.num_vector_core):
                old = TileGenerator.descend_split_tiling(reference)
            generator = ComboTileGenerator(copy.deepcopy(kernels))
            actual = generator.descend_split_tiling()
            key = lambda c: (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
            self.assertLessEqual({key(c) for c in old}, {key(c) for c in actual})
            self.assertEqual(len(actual), len({key(c) for c in actual}))
            self.assertEqual(generator.max_programs, config.num_vector_core * config.combo_kernel_max_waves)

    def test_infeasible_lower_wave_budgets_are_skipped(self):
        generator = ComboTileGenerator([subkernel(1, 1)] * (config.num_vector_core + 1))
        self.assertTrue(generator.descend_split_tiling())
        self.assertEqual(generator.max_programs, config.num_vector_core * config.combo_kernel_max_waves)

    def test_configured_wave_budgets_and_candidate_retention(self):
        previous = set()
        for waves in (1, 4, 8):
            with self.subTest(waves=waves), patch.object(config, 'combo_kernel_max_waves', waves):
                generator = ComboTileGenerator([subkernel(794624, 1), subkernel(3178496, 1)])
                limit = config.num_vector_core * waves
                self.assertEqual(generator.max_waves, waves)
                self.assertEqual(generator.max_programs, limit)
                candidates = generator.descend_split_tiling()
                key = lambda c: (tuple(sorted(c.kwargs.items())), c.num_warps, c.num_stages)
                actual = {key(c) for c in candidates}
                self.assertLessEqual(previous, actual)
                self.assertEqual(len(candidates), len(actual))
                self.assertEqual(generator.max_programs, limit)
                self.assertTrue(any(generator.calculate_total_programs(
                    [c.kwargs['X0BLOCK'], 1]) > config.num_vector_core * (waves - 1)
                    for c in candidates))
                self.assertTrue(all(generator.calculate_total_programs(
                    [c.kwargs['X0BLOCK'], 1]) <= limit for c in candidates))
                previous = actual

    def test_wave_budget_restored_when_search_fails(self):
        generator = ComboTileGenerator([subkernel(1000, 1)])
        limit = generator.max_programs
        with patch.object(TileGenerator, 'descend_split_tiling', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                generator.descend_split_tiling()
        self.assertEqual(generator.max_programs, limit)

    def test_mixed_dtype_footprint_is_branch_local(self):
        kernels = [subkernel(208, 788, True), subkernel(388, 20, True)]
        kernels[1]['split_axis_dtype'] = torch.int64
        generator = ComboTileGenerator(kernels, input_ptr_num=3)
        footprint = generator.calculate_config_numel({'X0BLOCK_SUB': 24})
        # The float32 branch is larger; normalize its footprint to int64 units.
        self.assertEqual(footprint, 24 * 788 // 2)

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

    def test_no_split_subkernel_still_has_one_program(self):
        item = subkernel(1)
        item['split_axis'] = []
        generator = ComboTileGenerator([item, copy.deepcopy(item)])
        self.assertEqual(generator.calculate_total_programs([1, 128]), 2)
        self.assertTrue(generator.descend_split_tiling())


if __name__ == '__main__':
    unittest.main()
