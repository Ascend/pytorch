import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFastGelu(TestCase):

    def supported_op_exec(self, input1):
        attr = 1.702
        attr_half = attr / 2
        abs_input1 = torch.abs(input1)
        numerator = input1 * torch.exp((attr_half * input1) * (input1 - abs_input1))
        denominator = 1.0 + torch.exp(- attr * abs_input1)
        output = numerator / denominator
        return output.cpu().detach()

    def custom_op_exec(self, input1):
        output = torch_npu.fast_gelu(input1)
        return output.cpu().detach()

    def test_fast_gelu(self, device="npu"):
        item = [np.float32, 0, [3, 16, 32]]
        _, npu_input = create_common_tensor(item, 0, 100)

        supported_output = self.supported_op_exec(npu_input)
        custom_output = self.custom_op_exec(npu_input)
        self.assertRtolEqual(supported_output, custom_output)

    def test_fast_gelu_input_arg(self):
        item = [np.float32, 0, [3, 16, 32]]
        _, npu_input = create_common_tensor(item, 0, 100)
        supported_output = self.supported_op_exec(npu_input)
        custom_output = torch_npu.fast_gelu(input=npu_input)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
