from torch.testing._internal.common_utils import TestCase, run_tests

import torch_npu


class TestJitCompile(TestCase):
    def test_jit_compile_default(self):
        # set_device initializes jitCompileInit without explicitly setting jitCompile.
        torch_npu.npu.set_device(0)
        self.assertTrue(torch_npu.npu.is_jit_compile_false())

    def test_jit_compile_false(self):
        torch_npu.npu.set_compile_mode(jit_compile=False)
        self.assertTrue(torch_npu.npu.is_jit_compile_false())

    def test_jit_compile_true(self):
        # set_device initializes jitCompileInit before jitCompile is explicitly enabled.
        torch_npu.npu.set_device(0)
        torch_npu.npu.set_compile_mode(jit_compile=True)
        self.assertFalse(torch_npu.npu.is_jit_compile_false())


if __name__ == "__main__":
    run_tests()
