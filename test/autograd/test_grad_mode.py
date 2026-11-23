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
Add validation cases for torch.autograd.grad_mode.inference_mode API on NPU:
1. PyTorch community lacks sufficient and direct API validations for this API, so this file is added.
2. This file validates torch.autograd.grad_mode.inference_mode (extendable).
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


def mode_flag():
    """Return the inference mode flag of the calling thread."""
    return torch.is_inference_mode_enabled()


def version_counter(tensor):
    """Read the version counter of a tensor; inference tensors do not track one."""
    return tensor._version


class TestInferenceMode(TestCase):
    def test_context_manager_enable_and_restore(self):
        """Test the context manager enables inference mode and restores the previous state."""
        self.assertFalse(mode_flag())
        with torch.autograd.grad_mode.inference_mode():
            self.assertTrue(mode_flag())
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(mode_flag())

    def test_context_manager_mode_argument(self):
        """Test the mode argument accepts positional, keyword, enabled and disabled values."""
        seen = []
        with torch.inference_mode(True):
            seen.append(mode_flag())
        with torch.inference_mode(mode=True):
            seen.append(mode_flag())
        with torch.inference_mode(False):
            seen.append(mode_flag())
        with torch.inference_mode(mode=False):
            seen.append(mode_flag())
        self.assertEqual(seen, [True, True, False, False])

    def test_nested_context_managers_restore_outer_state(self):
        """Test nested contexts with opposite modes restore the outer state after exit."""
        seen = [mode_flag()]
        with torch.autograd.grad_mode.inference_mode(True):
            seen.append(mode_flag())
            with torch.autograd.grad_mode.inference_mode(False):
                seen.append(mode_flag())
            seen.append(mode_flag())
        seen.append(mode_flag())
        self.assertEqual(seen, [False, True, False, True, False])

    def test_decorator_supported_forms(self):
        """Test the decorator works bare and with empty, positional and keyword arguments."""

        @torch.inference_mode
        def bare_form():
            return mode_flag()

        @torch.inference_mode()
        def empty_form():
            return mode_flag()

        @torch.inference_mode(False)
        def positional_form():
            return mode_flag()

        @torch.inference_mode(mode=False)
        def keyword_form():
            return mode_flag()

        self.assertEqual(bare_form(), True)
        self.assertEqual(empty_form(), True)
        self.assertEqual(positional_form(), False)
        self.assertEqual(keyword_form(), False)

    def test_decorator_accepts_callable_as_mode(self):
        """Test that a callable passed as mode gets decorated with inference mode enabled."""
        seen = []

        def weighted_sum(t):
            seen.append(mode_flag())
            return torch.add(t, t)

        wrapped = torch.inference_mode(weighted_sum)
        t = torch.arange(6, dtype=torch.float32, device=device_type).reshape(2, 3)
        result = wrapped(t)
        self.assertEqual(wrapped.__name__, weighted_sum.__name__)
        self.assertEqual(seen, [True])
        self.assertTrue(torch.is_inference(result))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.device.type, device_type)

    def test_inference_tensor_created_on_device(self):
        """Test tensors created in inference mode are inference tensors without version counter."""
        with torch.inference_mode():
            filled = torch.full((4, 3), 2.5, dtype=torch.float32, device=device_type)
            self.assertEqual(filled.device.type, device_type)
            self.assertFalse(filled.requires_grad)
            self.assertTrue(torch.is_inference(filled))
            with self.assertRaises(RuntimeError) as cm:
                version_counter(filled)
            self.assertIn("version counter", str(cm.exception))

    def test_tensor_created_with_mode_false(self):
        """Test tensors created with mode disabled keep normal autograd behaviour."""
        with torch.inference_mode(False):
            x = torch.full((2, 2), 3.0, device=device_type, requires_grad=True)
            self.assertFalse(torch.is_inference(x))
            self.assertTrue(x.requires_grad)
            self.assertEqual(version_counter(x), 0)

    def test_operation_in_inference_mode_on_device(self):
        """Test a non-view operation on an NPU tensor produces an inference result."""
        x = torch.full((2, 2), 3.0, device=device_type, requires_grad=True)
        with torch.inference_mode():
            y = torch.mul(x, x)
        self.assertEqual(y.device.type, device_type)
        self.assertFalse(y.requires_grad)
        self.assertTrue(torch.is_inference(y))

    def test_clone_preserves_mode(self):
        """Test clone returns an independent context manager that keeps the mode value."""
        for mode in (True, False):
            ctx = torch.autograd.grad_mode.inference_mode(mode)
            cloned = ctx.clone()
            self.assertIsNot(ctx, cloned)
            self.assertIsInstance(cloned, torch.autograd.grad_mode.inference_mode)
            self.assertEqual(cloned.mode, mode)

    def test_clone_context_manager_usable(self):
        """Test a cloned context manager can still control inference mode."""
        with torch.inference_mode().clone():
            self.assertTrue(mode_flag())
        with torch.inference_mode(False).clone():
            self.assertFalse(mode_flag())
        self.assertFalse(mode_flag())


if __name__ == "__main__":
    run_tests()
