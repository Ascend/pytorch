import unittest

import torch

import torch_npu
from torch_npu.utils.cpp_extension import BISHENG_CPP_HOME
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
    @unittest.skipIf(BISHENG_CPP_HOME is None, "BISHENG_CPP_HOME nor found")
    def test_bisheng_extension(self):
        import torch_test_cpp_extension.bisheng as bisheng_extension
        x = torch.randn(4, 4).npu()
        y = torch.randn(4, 4).npu()
        z = bisheng_extension.bscpp_add(x, y)
        self.assertEqual(z.cpu(), (x + y).cpu())

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

if __name__ == "__main__":
    run_tests()
