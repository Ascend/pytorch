import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMishBackward(TestCase):
    def npu_op_exec(self, input1):
        input1.requires_grad = True
        output = torch_npu.npu_mish(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.cpu().detach().numpy()
        return output_grad, output

    def cpu_op_exec(self, input1):
        input1.requires_grad = True
        output = input1 * (torch.tanh(F.softplus(input1)))
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output_grad, output

    def test_mish_fp32(self):
        npu_input = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]).npu()
        cpu_input = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        output_grad, npu_output = self.npu_op_exec(npu_input)
        ep_output_grad, ep_npu_output = self.cpu_op_exec(cpu_input)
        self.assertRtolEqual(ep_output_grad, output_grad, prec=3.e-4)
        self.assertRtolEqual(ep_npu_output, npu_output)


if __name__ == "__main__":
    run_tests()
