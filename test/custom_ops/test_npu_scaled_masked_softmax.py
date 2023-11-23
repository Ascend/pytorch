import unittest
import torch
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestScaledMaskedSoftmax(TestCase):
    def cpu_to_exec_fixed_triu_mask_true(self, x, mask, scale=1):
        mask_tri = torch.triu(torch.ones(
            mask.shape, device='npu'), diagonal=1).bool()
        mask_data = (x * scale).masked_fill(mask_tri, value=-1e4)
        output = torch.nn.functional.softmax(mask_data, dim=-1)
        return output

    def cpu_to_exec_fixed_triu_mask_false(self, x, mask, scale=1):
        mask_data = (x * scale).masked_fill(mask, value=-1e4)
        output = torch.nn.functional.softmax(mask_data, dim=-1)
        return output

    def npu_to_exec(self, x, mask, scale, fixed_triu_mask):
        output = torch_npu.npu_scaled_masked_softmax(
            x, mask, scale, fixed_triu_mask)
        return output

    @unittest.skip("skip test_scaled_masked_softmax_triu_false now")
    def test_scaled_masked_softmax_triu_false(self):
        x = torch.randn(16, 6, 128, 128, dtype=torch.float32).npu()
        mask = torch.randn(16, 6, 128, 128, dtype=torch.float32).npu()
        mask = mask > 0
        scale = 0.334
        fixed_triu_mask = False

        cpu_out = self.cpu_to_exec_fixed_triu_mask_false(x, mask, scale)
        npu_out = self.npu_to_exec(x, mask, scale, fixed_triu_mask)
        cpu_out = cpu_out.half().cpu().numpy()
        npu_out = npu_out.cpu().numpy()

        self.assertRtolEqual(cpu_out, npu_out)

    @unittest.skip("skip test_scaled_masked_softmax_triu_true now")
    def test_scaled_masked_softmax_triu_true(self):
        x = torch.randn(16, 6, 128, 128, dtype=torch.float32).npu()
        mask = torch.randn(16, 6, 128, 128, dtype=torch.float32).npu()
        mask = mask > 0
        scale = 0.56
        fixed_triu_mask = True

        cpu_out = self.cpu_to_exec_fixed_triu_mask_true(x, mask, scale)
        npu_out = self.npu_to_exec(x, mask, scale, fixed_triu_mask)
        cpu_out = cpu_out.half().cpu().numpy()
        npu_out = npu_out.cpu().numpy()

        self.assertRtolEqual(cpu_out, npu_out)


if __name__ == "__main__":
    run_tests()
