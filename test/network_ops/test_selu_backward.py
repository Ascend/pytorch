import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSeluBackward(TestCase):

    def cpu_op_exec(self, input_x, inplace):
        input_x.requires_grad = True
        selu_input = input_x + 1
        selu = torch.nn.SELU(inplace=inplace)
        output = selu(selu_input)
        loss = output.sum()
        loss.backward()
        return input_x.grad, output.detach()

    def npu_op_exec(self, input_x, inplace):
        input_x.requires_grad = True
        selu_input = input_x + 1
        selu = torch.nn.SELU(inplace=inplace)
        output = selu(selu_input)
        loss = output.sum()
        loss.backward()
        return input_x.grad.cpu(), output.detach().cpu()

    def test_selu_backward(self):
        dtype_list = [np.float32, np.float16]
        format_list = [0, 2, 3, 29]
        shape_list = [(2000), (64, 128), (64, 3, 128), (64, 3, 34, 128)]
        inplace_list = [True, False]
        shape_format = [
            [[i, j, k], l] for i in dtype_list for j in format_list for k in shape_list for l in inplace_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2.0, 2.0)
            cpu_grad, cpu_output = self.cpu_op_exec(cpu_input1.float(), item[1])
            if item[0][0] == np.float16:
                cpu_grad = cpu_grad.half()
                cpu_output = cpu_output.half()
            npu_grad, npu_output = self.npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_grad, npu_grad)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
