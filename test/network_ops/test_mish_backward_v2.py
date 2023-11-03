import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

option = {"ACL_OP_SELECT_IMPL_MODE": "high_precision", "ACL_OPTYPELIST_FOR_IMPLMODE": "MishGrad, Mish"}
torch.npu.set_option(option)


class TestMishBackward(TestCase):
    def npu_op_exec(self, input1):
        w = torch.ones_like(input1)
        input1.requires_grad = True
        mish = torch.nn.Mish()
        output = mish(input1)
        output.backward(w)
        output_grad = input1.grad.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        return output_grad, output

    def cpu_op_exec(self, input1):
        w = torch.ones_like(input1)
        input1.requires_grad = True
        mish = torch.nn.Mish()
        output = mish(input1)
        output.backward(w)
        output_grad = input1.grad.detach().numpy()
        output = output.detach().numpy()
        return output_grad, output

    def test_mish_fp32(self):
        shape_format = [
            [[np.float32, -1, [10, 30, 10]]],
            [[np.float32, 2, [20, 30, 20]]],
            [[np.float32, 29, [20, 40, 30]]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            npu_output_grad, npu_output = self.npu_op_exec(npu_input)
            cpu_output_grad, cpu_output = self.cpu_op_exec(cpu_input)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_mish_fp16(self):
        shape_format = [
            [[np.float16, -1, [10, 30, 10]]],
            [[np.float16, 2, [20, 30, 20]]],
            [[np.float16, 29, [20, 40, 30]]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            npu_output_grad, npu_output = self.npu_op_exec(npu_input)
            cpu_output_grad, cpu_output = self.cpu_op_exec(cpu_input.float())
            cpu_output_grad = cpu_output_grad.astype(np.float16)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
