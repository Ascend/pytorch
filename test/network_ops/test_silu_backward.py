import torch
import torch.nn.functional as F
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

input_grad = None
npu_input_grad = None


def input_grad_hook(grad):
    global input_grad
    input_grad = grad
    input_grad = input_grad.numpy()


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.to("cpu")
    npu_input_grad = npu_input_grad.numpy()


class TestSiluBackward(TestCase):
    def cpu_op_exec(self, input1, is_contiguous=True):
        if is_contiguous is False:
            input1 = input1.as_strided([2, 2], [1, 2], 1)
        input1.requires_grad = True
        input1.register_hook(input_grad_hook)
        output = F.silu(input1)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input1, is_contiguous=True):
        if is_contiguous is False:
            input1 = input1.as_strided([2, 2], [1, 2], 1)
        input1.requires_grad = True
        input1.register_hook(npu_input_grad_hook)

        output = F.silu(input1)
        z = output.sum()
        z.backward()
        input1 = input1.cpu()

    def test_silu_backward_shape_format_fp32(self):
        format_list = [0, 3, 4, 29]
        shape_list = [(256, 2048, 7, 7)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            input1, npu_input1 = create_common_tensor(item, 1, 100)
            input2, npu_input2 = create_common_tensor(item, 1, 100)
            self.cpu_op_exec(input1)
            self.npu_op_exec(npu_input1)
            self.assertRtolEqual(input_grad, npu_input_grad)

            self.cpu_op_exec(input2, False)
            self.npu_op_exec(npu_input2, False)
            self.assertRtolEqual(input_grad, npu_input_grad)

    def cpu_op_inplace_exec(self, x):
        x.requires_grad = True
        silu = torch.nn.SiLU(inplace=True)
        x1 = x + 0.1
        out = silu(x1)
        loss = out.mean()
        loss.backward()
        return out.detach(), x.grad

    def npu_op_inplace_exec(self, x):
        x.requires_grad = True
        silu = torch.nn.SiLU(inplace=True)
        x1 = x + 0.1
        out = silu(x1)
        loss = out.mean()
        loss.backward()
        return out.cpu().detach(), x.grad.cpu()

    def test_silu_backward_inplace(self):
        format_list = [0]
        shape_list = [(2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)

            cpu_out, cpu_grad = self.cpu_op_inplace_exec(cpu_input)
            npu_out, npu_grad = self.npu_op_inplace_exec(npu_input)
            self.assertRtolEqual(cpu_out, npu_out)
            self.assertRtolEqual(cpu_grad, npu_grad)


if __name__ == "__main__":
    run_tests()
