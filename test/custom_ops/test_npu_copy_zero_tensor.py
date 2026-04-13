import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestCopyZeroTensor(TestCase):
    """Test copying zero tensor to NPU tensor.

    This test verifies the fix for issue 146 where copying a CPU zero tensor
    (created by _efficientzerotensor) to NPU would fail with null pointer error
    in aclrtMemcpy. The fix checks _is_zerotensor() and calls zero_() instead.
    """

    def test_copy_zero_tensor_to_npu(self):
        """Test copying CPU zero tensor to NPU tensor."""
        zero_tensor = torch._efficientzerotensor((3, 4), dtype=torch.float32)

        self.assertTrue(zero_tensor._is_zerotensor())
        npu_tensor = torch.randn(3, 4).npu()
        npu_tensor.copy_(zero_tensor)

        expected = torch.zeros(3, 4)
        self.assertRtolEqual(npu_tensor.cpu().numpy(), expected.numpy())

    def test_copy_zero_tensor_various_shapes(self):
        """Test copying zero tensor with various shapes."""
        shapes = [(2, 3), (5,), (2, 3, 4), (10, 5)]

        for shape in shapes:
            zero_tensor = torch._efficientzerotensor(shape, dtype=torch.float32)
            npu_tensor = torch.ones(shape, dtype=torch.float32).npu()
            npu_tensor.copy_(zero_tensor)
            expected = torch.zeros(shape, dtype=torch.float32)
            self.assertRtolEqual(npu_tensor.cpu().numpy(), expected.numpy())

    def test_copy_zero_tensor_various_dtypes(self):
        """Test copying zero tensor with various dtypes."""
        dtypes = [torch.float32, torch.float16, torch.int32]
        shape = (3, 4)

        for dtype in dtypes:
            zero_tensor = torch._efficientzerotensor(shape, dtype=dtype)
            npu_tensor = torch.ones(shape, dtype=dtype).npu()
            npu_tensor.copy_(zero_tensor)
            expected = torch.zeros(shape, dtype=dtype)
            self.assertRtolEqual(npu_tensor.cpu().numpy(), expected.numpy())

    def test_copy_zero_tensor_from_functorch(self):
        """Test the original failing case from functorch test.

        This reproduces the scenario from test_linearize_composition_grad_npu_float32
        where _efficientzerotensor falls back to CPU and then gets copied to NPU.
        """
        shape = (2, 3)
        dtype = torch.float32
        zero_tensor = torch._efficientzerotensor(shape, dtype=dtype)
        npu_tensor = torch.randn(shape, dtype=dtype).npu()

        npu_tensor.copy_(zero_tensor)
        expected = torch.zeros(shape, dtype=dtype)
        self.assertRtolEqual(npu_tensor.cpu().numpy(), expected.numpy())


if __name__ == "__main__":
    run_tests()
