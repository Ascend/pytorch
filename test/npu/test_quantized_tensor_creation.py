"""
Add validation cases for quantized tensor creation APIs on NPU:

1. PyTorch community covers torch._empty_affine_quantized through view-op tests,
   but lacks an independently runnable NPU validation for this API.

2. This file validates the following APIs:
   torch._empty_affine_quantized (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
)


class TestQuantizedTensorCreation(TestCase):
    """Test quantized tensor creation APIs on NPU."""

    SCALE = 0.5
    ZERO_POINT = 3

    def _create_tensor(self, shape):
        return torch._empty_affine_quantized(
            shape,
            scale=self.SCALE,
            zero_point=self.ZERO_POINT,
            dtype=torch.quint8,
            device=device_type,
        )

    def test_empty_affine_quantized_properties(self):
        """Verify tensor properties and per-tensor affine quantization metadata."""
        tensor = self._create_tensor((2, 3, 4))

        self.assertEqual(tensor.shape, torch.Size([2, 3, 4]))
        self.assertEqual(tensor.numel(), 24)
        self.assertEqual(tensor.dtype, torch.quint8)
        self.assertEqual(tensor.device.type, device_type)
        self.assertEqual(tensor.qscheme(), torch.per_tensor_affine)
        self.assertEqual(tensor.q_scale(), self.SCALE)
        self.assertEqual(tensor.q_zero_point(), self.ZERO_POINT)

    def test_empty_affine_quantized_zero_numel(self):
        """Verify creation of quantized tensors containing a zero-sized dimension."""
        shapes = ((0, 2, 3), (3, 0, 2))

        for shape in shapes:
            with self.subTest(shape=shape):
                tensor = self._create_tensor(shape)

                self.assertEqual(tensor.shape, torch.Size(shape))
                self.assertEqual(tensor.numel(), 0)
                self.assertEqual(tensor.dtype, torch.quint8)
                self.assertEqual(tensor.device.type, device_type)
                self.assertEqual(tensor.qscheme(), torch.per_tensor_affine)
                self.assertEqual(tensor.q_scale(), self.SCALE)
                self.assertEqual(tensor.q_zero_point(), self.ZERO_POINT)

    def test_empty_affine_quantized_parameters(self):
        """Verify creation with different affine quantization parameters."""
        test_cases = (
            ((2, 3), 0.1, 0, torch.quint8),
            ((3, 4), 0.25, 5, torch.quint8),
            ((1, 2, 3), 1.0, -3, torch.qint8),
        )

        for shape, scale, zero_point, dtype in test_cases:
            with self.subTest(
                shape=shape,
                scale=scale,
                zero_point=zero_point,
                dtype=dtype,
            ):
                tensor = torch._empty_affine_quantized(
                    shape,
                    scale=scale,
                    zero_point=zero_point,
                    dtype=dtype,
                    device=device_type,
                )

                self.assertEqual(tensor.shape, torch.Size(shape))
                self.assertEqual(tensor.dtype, dtype)
                self.assertEqual(tensor.device.type, device_type)
                self.assertEqual(tensor.qscheme(), torch.per_tensor_affine)
                self.assertEqual(tensor.q_scale(), scale)
                self.assertEqual(tensor.q_zero_point(), zero_point)


if __name__ == "__main__":
    run_tests()