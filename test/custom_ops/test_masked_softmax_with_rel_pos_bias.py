import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestMaskedSoftmaxWithRelPosBias(TestCase):

    def supported_op_exec(self, x, relative_pos_bias, atten_mask):
        # add + add + softmax 
        y = torch.add(x, atten_mask)
        y = torch.add(y, relative_pos_bias)
        softmax_out = torch.nn.functional.softmax(y, dim=-1)
        return softmax_out.cpu()

    def custom_op_exec(self, x, relative_pos_bias, atten_mask):
        return torch_npu.npu_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias).cpu()

    @SupportedDevices(['Ascend910B'])
    def test_npu_masked_softmax_with_rel_pos_bias(self, device="npu"):
        x = torch.randn(1, 2, 3, 4, 4, dtype=torch.float)
        relative_pos_bias = torch.randn(1, 1, 3, 4, 4, dtype=torch.float)
        atten_mask = torch.randn(1, 2, 1, 4, 4, dtype=torch.float)

        supported_output = self.supported_op_exec(x, relative_pos_bias, atten_mask)
        custom_output = self.custom_op_exec(x.npu(), relative_pos_bias.npu(), atten_mask.npu())
        self.assertRtolEqual(supported_output, custom_output)

if __name__ == "__main__":
    run_tests()
