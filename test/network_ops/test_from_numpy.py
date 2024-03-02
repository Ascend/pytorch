import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFromNumpy(TestCase):

    def test_from_numpy_float16(self):
        npary = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float16)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.size(), torch.Size([2, 3]))

    def test_from_numpy_float32(self):
        npary = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.size(), torch.Size([2, 4]))

    def test_from_numpy_int64(self):
        npary = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.int64)
        self.assertEqual(output.size(), torch.Size([2, 4]))

    def test_from_numpy_int32(self):
        npary = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.int32)
        self.assertEqual(output.size(), torch.Size([2, 4]))

    def test_from_numpy_int16(self):
        npary = np.array([[1, 2, 3], [5, 6, 7]], dtype=np.int16)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.int16)
        self.assertEqual(output.size(), torch.Size([2, 3]))

    def test_from_numpy_bool(self):
        npary = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.bool_)
        output = torch.from_numpy(npary)
        self.assertEqual(output.dtype, torch.bool)
        self.assertEqual(output.size(), torch.Size([2, 4]))

if __name__ == "__main__":
    run_tests()
