import torch
from torch_npu.testing.testcase import TestCase, run_tests


class TorchBackendsApiTestCase(TestCase):

    def test_aclnn_allow_hf32(self):
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        torch.npu.aclnn.allow_hf32 = True
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, True)
        torch.npu.aclnn.allow_hf32 = False
        res = torch.npu.aclnn.allow_hf32
        self.assertEqual(res, False)

    def test_conv_allow_hf32(self):
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = True
        res = torch.npu.conv.allow_hf32
        self.assertEqual(res, True)
        torch.npu.conv.allow_hf32 = False
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
