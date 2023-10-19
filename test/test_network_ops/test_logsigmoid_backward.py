import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

cpu_input_grad = None
npu_input_grad = None


def cpu_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad.numpy()


def cpu_float16_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad.numpy()
    cpu_input_grad = cpu_input_grad.astype(np.float16)


def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.cpu().numpy()


class TestLogSigmoidBackward(TestCase):
    def cpu_op_exec(self, input1):
        input1.requires_grad = True
        input1.register_hook(cpu_input_grad_hook)
        output = torch.nn.functional.logsigmoid(input1)
        z = output.sum()
        z.backward()

    def npu_op_exec(self, input1):
        input1.requires_grad = True
        input1.register_hook(npu_input_grad_hook)
        output = torch.nn.functional.logsigmoid(input1)
        z = output.sum()
        z.backward()

    def test_log_sigmoid_backward_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (6, 4)]],
            [[np.float32, 3, (2, 4, 5)]],
            [[np.float32, 4, (1, 2, 3, 3)]],
            [[np.float32, 29, (10, 3, 5, 3)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -50, 50)
            self.cpu_op_exec(cpu_input)
            self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_log_sigmoid_backward_float16_shape_format(self, device="npu"):
        def cpu_op_exec_fp16(input1):
            input1.requires_grad = True
            input1.register_hook(cpu_float16_input_grad_hook)
            input1 = input1.to(torch.float32)
            output = torch.nn.functional.logsigmoid(input1)
            z = output.sum()
            z.backward()

        shape_format = [
            [[np.float16, 0, (6, 4)]],
            [[np.float16, 3, (2, 4, 5)]],
            [[np.float16, 4, (1, 2, 3, 3)]],
            [[np.float16, 29, (10, 3, 5, 3)]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -50, 50)
            cpu_op_exec_fp16(cpu_input1)
            self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_input_grad, npu_input_grad)


if __name__ == "__main__":
    run_tests()
