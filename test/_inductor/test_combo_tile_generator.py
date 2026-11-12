"""Arithmetic checks for combo split boundaries; NPU accuracy is tested separately."""
import copy
import unittest

import torch

from torch_npu._inductor import config
from torch_npu._inductor.runtime.combo_tile_generator import ComboTileGenerator


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
    def test_candidates_fit_one_wave_and_ignore_subkernel_order(self):
        kernels = [subkernel(x) for x in (555, 423, 111)]
        original = copy.deepcopy(kernels)
        first = ComboTileGenerator(kernels)
        candidates = first.descend_split_tiling()
        self.assertTrue(candidates)
        for candidate in candidates:
            block = candidate.kwargs['X0BLOCK']
            self.assertLessEqual(first.calculate_total_programs([block, 128]),
                                 config.num_vector_core)
            self.assertIn('X1BLOCK_SUB', candidate.kwargs)
        second = ComboTileGenerator(list(reversed(kernels)))
        other = second.descend_split_tiling()

        def normalize(values):
            return {tuple(sorted(item.kwargs.items())) for item in values}

        self.assertEqual(normalize(candidates), normalize(other))
        self.assertEqual(kernels, original)

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
