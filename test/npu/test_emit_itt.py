# Copyright (c) 2026 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Add validation cases for torch.autograd.profiler.emit_itt on Ascend NPU.

Intel ITT is not available on some NPU build configurations.
This file validates emit_itt parameter coverage (enabled, record_shapes).
Tests with enabled=True are guarded by ITT availability check,
matching the upstream test_profiler_emit_itt pattern.
"""
import torch
from torch.autograd.profiler import emit_itt

from torch_npu.testing.testcase import TestCase, run_tests


ITT_AVAILABLE = torch.profiler.itt.is_available()


class TestEmitItt(TestCase):
    """Test cases for torch.autograd.profiler.emit_itt on Ascend NPU."""

    def test_emit_itt_import(self):
        """emit_itt is callable or context manager."""
        self.assertTrue(callable(emit_itt) or hasattr(emit_itt, "__enter__"))

    def test_emit_itt_enabled_false_noop(self):
        """enabled=False is a no-op and computation works normally."""
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt(enabled=False):
            result = x + 1.0
        expected = torch.tensor([2.0, 3.0, 4.0], device="npu")
        self.assertTrue(torch.equal(result, expected))

    def test_emit_itt_enabled_false_record_shapes_true(self):
        """enabled=False + record_shapes=True: no-op with correct result."""
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt(enabled=False, record_shapes=True):
            result = x - 1.0
        expected = torch.tensor([0.0, 1.0, 2.0], device="npu")
        self.assertTrue(torch.equal(result, expected))

    def test_emit_itt_enabled_true_record_shapes_true(self):
        """enabled=True + record_shapes=True (guarded by ITT availability)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt(enabled=True, record_shapes=True):
            result = x - 1.0
        expected = torch.tensor([0.0, 1.0, 2.0], device="npu")
        self.assertTrue(torch.equal(result, expected))

    def test_emit_itt_enabled_true_record_shapes_false(self):
        """enabled=True + record_shapes=False (guarded by ITT availability)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt(enabled=True, record_shapes=False):
            result = x * 2.0
        expected = torch.tensor([2.0, 4.0, 6.0], device="npu")
        self.assertTrue(torch.equal(result, expected))

    def test_emit_itt_enabled_true_default(self):
        """enabled=True (default) does not disrupt computation (guarded)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt():
            result = x + 1.0
        expected = torch.tensor([2.0, 3.0, 4.0], device="npu")
        self.assertTrue(torch.equal(result, expected))

    def test_emit_itt_with_model_npu(self):
        """emit_itt works with NN model on NPU (guarded by ITT availability)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        model = torch.nn.Linear(10, 5).npu()
        x = torch.randn(3, 10, device="npu")
        with emit_itt():
            output = model(x)
        self.assertEqual(output.shape, (3, 5))

    def test_emit_itt_execution_order(self):
        """Operations inside emit_itt execute in correct order (guarded)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt():
            x = x + 1.0
            x = x * 2.0
        expected = torch.tensor([4.0, 6.0, 8.0], device="npu")
        self.assertTrue(torch.equal(x, expected))

    def test_emit_itt_record_shapes_true(self):
        """record_shapes=True does not disrupt computation (guarded)."""
        if not ITT_AVAILABLE:
            self.skipTest("ITT is required")
        torch.manual_seed(42)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device="npu")
        with emit_itt(record_shapes=True):
            result = x * 2.0
        expected = torch.tensor([2.0, 4.0, 6.0], device="npu")
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    run_tests()
