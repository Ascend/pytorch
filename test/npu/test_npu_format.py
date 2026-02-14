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
        x1 = torch.ones(2, 2).npu()
        torch_npu.npu_format_cast_(x1, torch_npu.Format.FRACTAL_NZ)

        weak_x1 = _weak_ref_tensor(x1)
        fmt1 = torch_npu.get_npu_format(weak_x1)
        self.assertEqual(fmt1, torch_npu.Format.FRACTAL_NZ)
        self.assertEqual(x1.data_ptr(), weak_x1.data_ptr())


if __name__ == "__main__":
    run_tests()
