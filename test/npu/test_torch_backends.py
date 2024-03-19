import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TorchBackendsApiTestCase(TestCase):

    def test_flash_sdp(self):
        torch.npu.enable_flash_sdp(True)
        res = torch.npu.flash_sdp_enabled()
        self.assertEqual(res, True)
        torch.npu.enable_flash_sdp(False)
        res = torch.npu.flash_sdp_enabled()
        self.assertEqual(res, False)

    def test_mem_efficient_sdp(self):
        torch.npu.enable_mem_efficient_sdp(False)
        res = torch.npu.mem_efficient_sdp_enabled()
        self.assertEqual(res, False)
        torch.npu.enable_mem_efficient_sdp(True)
        res = torch.npu.mem_efficient_sdp_enabled()
        self.assertEqual(res, True)

    def test_math_sdp(self):
        torch.npu.enable_math_sdp(True)
        res = torch.npu.math_sdp_enabled()
        self.assertEqual(res, True)
        torch.npu.enable_math_sdp(False)
        res = torch.npu.math_sdp_enabled()
        self.assertEqual(res, False)

    def test_sdp_kernel(self):
        torch.npu.enable_flash_sdp(False)
        torch.npu.enable_mem_efficient_sdp(False)
        torch.npu.enable_math_sdp(False)
        torch.npu.sdp_kernel()
        flash_res = torch.npu.flash_sdp_enabled()
        mem_mem_efficient_res = torch.npu.mem_efficient_sdp_enabled()
        math_res = torch.npu.math_sdp_enabled()
        self.assertEqual(flash_res, False)
        self.assertEqual(mem_mem_efficient_res, False)
        self.assertEqual(math_res, False)

    def test_aclnn_allow_hf32(self):
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=True):
            res = torch.npu.aclnn.allow_hf32
            self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=False):
            res = torch.npu.aclnn.allow_hf32
            self.assertEqual(res, False)

    def test_conv_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = False
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, False)

    def test_aclnn_to_conv_allow_hf32(self):
        torch.npu.conv.allow_hf32 = True
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = False
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, False)
        with torch.npu.aclnn.flags(allow_hf32=True):
            res = torch.npu.conv.allow_hf32
            self.assertEqual(res, True)
        with torch.npu.aclnn.flags(allow_hf32=False):
            res = torch.npu.conv.allow_hf32
            self.assertEqual(res, False)

    def test_matmul_allow_hf32(self):
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, False)
        torch.npu.matmul.allow_hf32 = True
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, True)
        torch.npu.matmul.allow_hf32 = False
        res = torch.npu.matmul.allow_hf32
        self.assertEqual(res, False)

    def test_preferred_linalg_library(self):
        res = torch.npu.preferred_linalg_library()
        self.assertEqual(res, torch._C._LinalgBackend.Default)
        res = torch.npu.preferred_linalg_library("Cusolver")
        self.assertEqual(res, torch._C._LinalgBackend.Default)
        res = torch.npu.preferred_linalg_library(torch._C._LinalgBackend.Magma)
        self.assertEqual(res, torch._C._LinalgBackend.Default)


if __name__ == "__main__":
    run_tests()
