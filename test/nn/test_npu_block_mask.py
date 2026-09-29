# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.nn.attention.flex_attention APIs on NPU:
1. PyTorch community lacks direct and sufficient API validations for BlockMask.to_string,
   create_mask, and create_nested_block_mask, so this file is added.
2. This file validates BlockMask.to_string with various grid_size and limit parameters.
3. This file validates create_mask with mask_mod and score_mod functions, various shapes.
4. This file validates create_nested_block_mask with nested tensors (available in PyTorch 2.7 only).
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention import flex_attention as flex_attention_module
from torch.testing._internal.common_utils import run_tests, TestCase


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
_has_create_nested_block_mask = hasattr(flex_attention_module, "create_nested_block_mask")


class TestBlockMaskToString(TestCase):
    def _create_causal_block_mask(self, B=2, H=2, Q_LEN=128, KV_LEN=128):
        """Create a causal block mask on NPU for testing."""
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        return create_block_mask(causal_mask, B, H, Q_LEN, KV_LEN, device=device_type)

    def test_to_string_default(self):
        """Verifies to_string with default parameters returns a valid string."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string()
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_to_string_grid_size_int(self):
        """Verifies to_string with grid_size as int."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string(grid_size=10)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_to_string_grid_size_tuple(self):
        """Verifies to_string with grid_size as tuple."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string(grid_size=(10, 10))
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_to_string_grid_size_non_square_tuple(self):
        """Verifies to_string with non-square grid_size tuple."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string(grid_size=(10, 20))
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_to_string_grid_size_negative_one(self):
        """Verifies to_string with grid_size=-1 (uncompressed)."""
        block_mask = self._create_causal_block_mask(Q_LEN=32, KV_LEN=32)
        result = block_mask.to_string(grid_size=-1)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_to_string_limit_zero(self):
        """Verifies to_string with limit=0 shows only truncation message."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string(limit=0)
        self.assertIsInstance(result, str)
        self.assertIn("...", result)

    def test_to_string_limit_one(self):
        """Verifies to_string with limit=1 shows one batch then truncates."""
        block_mask = self._create_causal_block_mask(B=4, H=2)
        result = block_mask.to_string(limit=1)
        self.assertIsInstance(result, str)
        self.assertIn("...", result)

    def test_to_string_limit_large(self):
        """Verifies to_string with limit larger than batch*head shows all."""
        block_mask = self._create_causal_block_mask(B=2, H=2)
        result = block_mask.to_string(limit=10)
        self.assertIsInstance(result, str)
        self.assertNotIn("...", result)

    def test_to_string_combined_params(self):
        """Verifies to_string with both grid_size and limit set."""
        block_mask = self._create_causal_block_mask(B=4, H=2)
        result = block_mask.to_string(grid_size=8, limit=2)
        self.assertIsInstance(result, str)
        self.assertIn("...", result)

    def test_to_string_contains_visual_chars(self):
        """Verifies to_string output contains visual representation characters."""
        block_mask = self._create_causal_block_mask()
        result = block_mask.to_string()
        self.assertIn("█", result)
        self.assertIn(" ", result)

    def test_to_string_contains_batch_indices(self):
        """Verifies to_string output contains batch index tuples."""
        block_mask = self._create_causal_block_mask(B=2, H=2)
        result = block_mask.to_string()
        self.assertIn("(0, 0)", result)

    def test_to_string_int_equals_tuple(self):
        """Verifies grid_size as int equals grid_size as tuple with same values."""
        block_mask = self._create_causal_block_mask()
        result_int = block_mask.to_string(grid_size=10)
        result_tuple = block_mask.to_string(grid_size=(10, 10))
        self.assertEqual(result_int, result_tuple)


class TestCreateMask(TestCase):
    @staticmethod
    def _causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @staticmethod
    def _causal_score_mod(score, b, h, q_idx, kv_idx):
        return torch.where(q_idx >= kv_idx, score, float("-inf"))

    def test_create_mask_basic_mask_mod(self):
        """Verifies create_mask with a mask_mod function returns correct shape and dtype."""
        mask = create_mask(self._causal_mask_mod, 2, 2, 128, 128, device=device_type)
        self.assertEqual(mask.shape, (2, 2, 128, 128))
        self.assertEqual(mask.dtype, torch.bool)

    def test_create_mask_basic_score_mod(self):
        """Verifies create_mask with a score_mod function returns correct shape."""
        mask = create_mask(self._causal_score_mod, 1, 1, 64, 64, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 64, 64))

    def test_create_mask_b_none(self):
        """Verifies create_mask with B=None defaults to 1."""
        mask = create_mask(self._causal_mask_mod, None, 1, 32, 32, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 32, 32))

    def test_create_mask_h_none(self):
        """Verifies create_mask with H=None defaults to 1."""
        mask = create_mask(self._causal_mask_mod, 1, None, 32, 32, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 32, 32))

    def test_create_mask_both_none(self):
        """Verifies create_mask with B=None and H=None defaults to 1."""
        mask = create_mask(self._causal_mask_mod, None, None, 16, 16, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 16, 16))

    def test_create_mask_different_q_kv_len(self):
        """Verifies create_mask with different Q_LEN and KV_LEN."""
        mask = create_mask(self._causal_mask_mod, 1, 1, 64, 128, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 64, 128))

    def test_create_mask_causal_values(self):
        """Verifies create_mask causal mask values are correct."""
        mask = create_mask(self._causal_mask_mod, 1, 1, 8, 8, device=device_type)
        self.assertTrue(mask[0, 0, 0, 0].item())
        self.assertTrue(mask[0, 0, 4, 4].item())
        self.assertFalse(mask[0, 0, 0, 4].item())
        self.assertTrue(mask[0, 0, 4, 0].item())

    def test_create_mask_device(self):
        """Verifies create_mask output is on the correct device."""
        mask = create_mask(self._causal_mask_mod, 1, 1, 16, 16, device=device_type)
        self.assertEqual(mask.device.type, device_type)

    def test_create_mask_prefix_lm(self):
        """Verifies create_mask with a prefix LM mask_mod."""
        def prefix_lm_mask(b, h, q_idx, kv_idx):
            prefix_len = 4
            return (q_idx >= kv_idx) | (kv_idx < prefix_len)

        mask = create_mask(prefix_lm_mask, 1, 1, 16, 16, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 16, 16))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertTrue(mask[0, 0, 0, 0].item())
        self.assertTrue(mask[0, 0, 0, 3].item())
        self.assertFalse(mask[0, 0, 0, 5].item())

    def test_create_mask_sliding_window(self):
        """Verifies create_mask with a sliding window mask_mod."""
        def sliding_window_mask(b, h, q_idx, kv_idx):
            window_size = 3
            return (q_idx - kv_idx < window_size) & (q_idx - kv_idx >= 0)

        mask = create_mask(sliding_window_mask, 1, 1, 16, 16, device=device_type)
        self.assertEqual(mask.shape, (1, 1, 16, 16))
        self.assertTrue(mask[0, 0, 5, 5].item())
        self.assertTrue(mask[0, 0, 5, 3].item())
        self.assertFalse(mask[0, 0, 5, 0].item())

    def test_create_mask_multi_batch_heads(self):
        """Verifies create_mask with multiple batches and heads."""
        mask = create_mask(self._causal_mask_mod, 4, 2, 32, 32, device=device_type)
        self.assertEqual(mask.shape, (4, 2, 32, 32))
        self.assertTrue(mask[0, 0, 0, 0].item())
        self.assertTrue(mask[3, 1, 16, 16].item())
        self.assertFalse(mask[3, 1, 0, 31].item())


class TestCreateNestedBlockMask(TestCase):
    def setUp(self):
        if not _has_create_nested_block_mask:
            self.skipTest("create_nested_block_mask is only available in PyTorch 2.7")

    @staticmethod
    def _causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def _create_nested_tensor(self, B=2, H=1, D=64):
        """Create a jagged layout nested tensor for testing."""
        seq_lengths = torch.tensor([32, 48], device=device_type)
        total_len = seq_lengths.sum().item()
        values = torch.randn(B, H, total_len, D, device=device_type)
        nt = torch.nested.nested_tensor(
            list(values.unbind(0)),
            layout=torch.jagged,
        )
        return nt

    def test_create_nested_block_mask_basic(self):
        """Verifies create_nested_block_mask returns a BlockMask."""
        create_nested_block_mask = flex_attention_module.create_nested_block_mask
        q_nt = self._create_nested_tensor(B=1, H=1, D=64)
        block_mask = create_nested_block_mask(
            self._causal_mask_mod, 1, 1, q_nt
        )
        self.assertIsInstance(block_mask, flex_attention_module.BlockMask)

    def test_create_nested_block_mask_with_kv_nt(self):
        """Verifies create_nested_block_mask with explicit kv_nt for cross attention."""
        create_nested_block_mask = flex_attention_module.create_nested_block_mask
        q_nt = self._create_nested_tensor(B=1, H=1, D=64)
        kv_nt = self._create_nested_tensor(B=1, H=1, D=64)
        block_mask = create_nested_block_mask(
            self._causal_mask_mod, 1, 1, q_nt, kv_nt
        )
        self.assertIsInstance(block_mask, flex_attention_module.BlockMask)

    def test_create_nested_block_mask_block_size(self):
        """Verifies create_nested_block_mask with custom BLOCK_SIZE."""
        create_nested_block_mask = flex_attention_module.create_nested_block_mask
        q_nt = self._create_nested_tensor(B=1, H=1, D=64)
        block_mask = create_nested_block_mask(
            self._causal_mask_mod, 1, 1, q_nt, BLOCK_SIZE=64
        )
        self.assertIsInstance(block_mask, flex_attention_module.BlockMask)
        self.assertEqual(block_mask.BLOCK_SIZE[0], 64)
        self.assertEqual(block_mask.BLOCK_SIZE[1], 64)

    def test_create_nested_block_mask_multi_batch(self):
        """Verifies create_nested_block_mask with multiple batches and heads."""
        create_nested_block_mask = flex_attention_module.create_nested_block_mask
        q_nt = self._create_nested_tensor(B=2, H=1, D=64)
        block_mask = create_nested_block_mask(
            self._causal_mask_mod, 2, 2, q_nt
        )
        self.assertIsInstance(block_mask, flex_attention_module.BlockMask)

    def test_create_nested_block_mask_device(self):
        """Verifies create_nested_block_mask output is on the correct device."""
        create_nested_block_mask = flex_attention_module.create_nested_block_mask
        q_nt = self._create_nested_tensor(B=1, H=1, D=64)
        block_mask = create_nested_block_mask(
            self._causal_mask_mod, 1, 1, q_nt
        )
        self.assertEqual(block_mask.kv_num_blocks.device.type, device_type)


if __name__ == "__main__":
    run_tests()
