import os
import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

try:
    import torch_test_cpp_extension.npu as npu_extension
except ImportError as e:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_cpp_test.py` instead.") from e


class TestCppExtensionAOT(TestCase):
    """Tests ahead-of-time cpp extensions
    """

    def test_npu_extension(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = npu_extension.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        z = npu_extension.tanh_add(x.npu(), y.npu())
        expect_out = x.npu().tanh() + y.npu().tanh()
        self.assertEqual(z.cpu(), expect_out.cpu())

        npu_z = npu_extension.npu_add(x.npu(), y.npu())
        self.assertEqual(npu_z.cpu(), (x + y))

    def test_storage_sizes(self):
        os.environ['INT8_FORMAT_NZ_ENABLE'] = '1'
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.int8).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (16, 8, 16, 32)))
        t = torch_npu.npu_format_cast(torch.ones(31, 127, 511, dtype=torch.int8).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (31, 16, 8, 16, 32)))
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.float16).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (32, 8, 16, 16)))
        # float32 will cast to float16 before calculate
        t = torch_npu.npu_format_cast(torch.ones(128, 512, dtype=torch.float32).npu(), 29)
        self.assertTrue(npu_extension.check_storage_sizes(t, (32, 8, 16, 16)))

    def test_from_blob(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self.assertTrue(npu_extension.check_from_blob())
        self.assertTrue(npu_extension.check_from_blob_strides())

if __name__ == "__main__":
    run_tests()
