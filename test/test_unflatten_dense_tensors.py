"""
Add validation cases for torch._utils APIs on NPU:
1. PyTorch community lacks direct Python unit tests for torch._utils._unflatten_dense_tensors.
2. This file validates torch._utils._unflatten_dense_tensors (extendable).
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


class TestUnflattenDenseTensors(TestCase):
    """Test cases for torch._utils._unflatten_dense_tensors."""

    def setUp(self):
        super().setUp()
        acc = torch.accelerator.current_accelerator()
        self.device = acc.type if acc else "cpu"

    def _to_device(self, t):
        return t.to(self.device)

    def test_round_trip_basic(self):
        # Round-trip: flatten then unflatten should recover original tensors
        t1 = self._to_device(torch.ones(4, 4))
        t2 = self._to_device(torch.zeros(2, 3))
        t3 = self._to_device(torch.randn(5))
        tensors = [t1, t2, t3]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), len(tensors))
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_single_tensor(self):
        t = self._to_device(torch.randn(3, 5, 2))
        flat = torch._utils._flatten_dense_tensors([t])
        result = torch._utils._unflatten_dense_tensors(flat, [t])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, t.shape)
        self.assertEqual(result[0], t)

    def test_multiple_tensors_different_sizes(self):
        sizes = [(1,), (2, 3), (4, 5, 6), (7, 8)]
        tensors = [self._to_device(torch.randn(*s)) for s in sizes]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), len(tensors))
        for i, (r, t) in enumerate(zip(result, tensors)):
            self.assertEqual(r.shape, t.shape, f"tensor {i} shape mismatch")
            self.assertEqual(r, t, f"tensor {i} value mismatch")

    def test_empty_tensor_in_list(self):
        t1 = self._to_device(torch.ones(3, 2))
        t2 = self._to_device(torch.tensor([]))
        t3 = self._to_device(torch.randn(4))
        tensors = [t1, t2, t3]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].numel(), 6)
        self.assertEqual(result[1].numel(), 0)
        self.assertEqual(result[2].numel(), 4)
        self.assertEqual(result[0], t1)
        # Verify empty tensor preserves shape, dtype and device
        self.assertEqual(result[1].shape, t2.shape)
        self.assertEqual(result[1].dtype, t2.dtype)
        self.assertEqual(result[1], t2)
        self.assertEqual(result[2], t3)

    def test_all_empty_tensors(self):
        tensors = [
            self._to_device(torch.tensor([])),
            self._to_device(torch.tensor([])),
        ]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 2)
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_different_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.int32]:
            with self.subTest(dtype=dtype):
                t1 = self._to_device(torch.ones(2, 3, dtype=dtype))
                t2 = self._to_device(torch.zeros(4, dtype=dtype))

                flat = torch._utils._flatten_dense_tensors([t1, t2])
                result = torch._utils._unflatten_dense_tensors(flat, [t1, t2])

                self.assertEqual(result[0].dtype, dtype)
                self.assertEqual(result[1].dtype, dtype)
                self.assertEqual(result[0], t1)
                self.assertEqual(result[1], t2)

    def test_large_num_tensors(self):
        n = 50
        tensors = [self._to_device(torch.randn(i + 1)) for i in range(n)]

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), n)
        for r, t in zip(result, tensors):
            self.assertEqual(r.shape, t.shape)
            self.assertEqual(r, t)

    def test_tuple_input(self):
        t1 = self._to_device(torch.ones(3, 3))
        t2 = self._to_device(torch.zeros(2))
        tensors = (t1, t2)  # tuple, not list

        flat = torch._utils._flatten_dense_tensors(tensors)
        result = torch._utils._unflatten_dense_tensors(flat, tensors)

        self.assertEqual(len(result), 2)
        for r, t in zip(result, tensors):
            self.assertEqual(r, t)


if __name__ == "__main__":
    run_tests()
