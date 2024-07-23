import expecttest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu
import torch_npu._C


class TestJitCompile(TestCase):
    def test_jit_compile_false(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self.assertTrue(torch.npu.is_jit_compile_false())

    def test_jit_compile_true(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self.assertFalse(torch.npu.is_jit_compile_false())


if __name__ == "__main__":
    run_tests()
