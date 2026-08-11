# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
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
Add validation cases for torch._functorch.config.patch API on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
2. This file validates torch._functorch.config.patch (extendable).
"""
import torch
import torch._functorch.config as config
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestFunctorchConfigPatch(TestCase):
    def test_basic_patch(self):
        """Test basic config.patch context manager"""
        original_value = config.debug_assert
        with config.patch("debug_assert", not original_value):
            self.assertEqual(config.debug_assert, not original_value)
        self.assertEqual(config.debug_assert, original_value)

    def test_patch_dict(self):
        """Test config.patch with dict argument"""
        original_debug = config.debug_assert
        original_cse = config.cse
        patches = {
            "debug_assert": not original_debug,
            "cse": not original_cse,
        }
        with config.patch(patches):
            self.assertEqual(config.debug_assert, not original_debug)
            self.assertEqual(config.cse, not original_cse)
        self.assertEqual(config.debug_assert, original_debug)
        self.assertEqual(config.cse, original_cse)

    def test_patch_restore_after_exception(self):
        """Test that config is restored even after an exception"""
        original_value = config.debug_assert
        with self.assertRaises(RuntimeError):
            with config.patch("debug_assert", not original_value):
                self.assertEqual(config.debug_assert, not original_value)
                raise RuntimeError("test exception")
        self.assertEqual(config.debug_assert, original_value)

    def test_patch_nested(self):
        """Test nested config.patch contexts"""
        original_value = config.debug_assert
        # First level
        with config.patch("debug_assert", True):
            self.assertTrue(config.debug_assert)
            # Second level (nested)
            with config.patch("debug_assert", False):
                self.assertFalse(config.debug_assert)
            # Back to first level
            self.assertTrue(config.debug_assert)
        # Back to original
        self.assertEqual(config.debug_assert, original_value)

    def test_patch_with_tensor_device(self):
        """Test that config.patch works correctly with NPU tensors in the context"""
        x = torch.randn(3, 4).to(device_type)
        original_value = config.debug_assert
        with config.patch("debug_assert", not original_value):
            # Verify NPU tensor operations still work
            y = x.mm(x.T)
            self.assertEqual(y.device.type, device_type)
            self.assertEqual(config.debug_assert, not original_value)
        self.assertEqual(config.debug_assert, original_value)

    def test_patch_invalid_key(self):
        """Test patch with invalid key raises AttributeError."""
        with self.assertRaises(AttributeError):
            with config.patch("non_existent_key", 42):
                pass

    def test_patch_invalid_dict_key(self):
        """Test patch dict with invalid key among valid keys raises AttributeError and does not change valid keys."""
        original_value = config.debug_assert
        with self.assertRaises(AttributeError):
            with config.patch({"debug_assert": True, "invalid_key": 42}):
                pass
        # Verify valid key is unchanged after the failed patch
        self.assertEqual(config.debug_assert, original_value)


if __name__ == "__main__":
    run_tests()
