import unittest
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu._C import _weak_ref_tensor


class TestNPUFormat(TestCase):
    
    def test_enum_values(self):
        """test the enumeration value"""
        self.assertEqual(torch_npu.Format.NCHW.value, 0)
        self.assertEqual(torch_npu.Format.NHWC.value, 1)

    def test_npu_format_cast(self):
        """test npu_format_cast"""
        tensor = torch.ones(2, 2).npu()

        out1 = torch_npu.npu_format_cast(tensor, 0)
        fmt1 = torch_npu.get_npu_format(out1)
        self.assertEqual(fmt1, torch_npu.Format.NCHW)

        out2 = torch_npu.npu_format_cast(tensor, torch_npu.Format.NHWC)
        fmt2 = torch_npu.get_npu_format(out2)
        self.assertEqual(fmt2, torch_npu.Format.NHWC)

    def test_npu_format_cast_(self):
        """test npu_format_cast_"""
        x1 = torch.ones(2, 2).npu()
        x2 = torch.ones(2, 2).npu()

        torch_npu.npu_format_cast_(x1, 0)
        fmt1 = torch_npu.get_npu_format(x1)
        self.assertEqual(fmt1, torch_npu.Format.NCHW)

        torch_npu.npu_format_cast_(x2, torch_npu.Format.NHWC)
        fmt2 = torch_npu.get_npu_format(x2)
        self.assertEqual(fmt2, torch_npu.Format.NHWC)

    def test_get_npu_format(self):
        """test get_npu_format"""
        x1 = torch.ones(2, 2).npu()
        torch_npu.npu_format_cast_(x1, 0)

        fmt1 = torch_npu.get_npu_format(x1)
        self.assertEqual(fmt1, torch_npu.Format.NCHW)
        self.assertEqual(fmt1, 0)

    @unittest.skip("Temporarily disabled")
    def test_get_npu_format_weak_ref(self):
        """test get_npu_format"""
        torch_npu.npu.config.allow_internal_format = True
        x1 = torch.ones(2, 2).npu()
        torch_npu.npu_format_cast_(x1, torch_npu.Format.FRACTAL_NZ)

        weak_x1 = _weak_ref_tensor(x1)
        fmt1 = torch_npu.get_npu_format(weak_x1)
        self.assertEqual(fmt1, torch_npu.Format.FRACTAL_NZ)
        self.assertEqual(x1.data_ptr(), weak_x1.data_ptr())

    def test_weak_ref_tensor_with_storage_offset(self):
        """test _weak_ref_tensor preserves shape, strides, offset and data"""
        view_shape = [2, 1, 8, 64]
        view_strides = [1536, 0, 192, 1]
        view_offset = 128

        max_offset = view_offset
        for i in range(len(view_shape)):
            max_offset += (view_shape[i] - 1) * view_strides[i]
        storage_size = max_offset + 1

        base = torch.arange(storage_size, dtype=torch.float32).npu()
        view = torch.as_strided(base, size=view_shape, stride=view_strides,
                                storage_offset=view_offset)

        weak = _weak_ref_tensor(view)
        self.assertEqual(weak.size(), view.size())
        self.assertEqual(weak.stride(), view.stride())
        self.assertEqual(weak.storage_offset(), view.storage_offset())
        self.assertEqual(weak.storage().nbytes(), view.storage().nbytes())
        self.assertTrue(torch.equal(weak, view))


if __name__ == "__main__":
    run_tests()
