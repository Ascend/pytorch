import expecttest

import torch
import torch_npu
import torch_npu._C

from torch_npu.testing._testcase import TestCase, run_tests


class TestJitCompile(TestCase):
    def test_jit_compile_false(self):
        torch.npu.set_compile_mode(jit_compile=False)
        self.assertTrue(torch.npu.is_jit_compile_false())

    def test_jit_compile_true(self):
        torch.npu.set_compile_mode(jit_compile=True)
        self.assertFalse(torch.npu.is_jit_compile_false())


if __name__ == "__main__":
    run_tests()
