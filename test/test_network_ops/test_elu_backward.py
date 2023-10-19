import torch
import numpy as np
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestEluBackward(TestCase):
    def cpu_op_exec(self, input1):
        flag = 0
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = 1
        input1.requires_grad = True
        output = torch.nn.functional.elu(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        if flag:
            output_grad = output_grad.to(torch.float16)
            output = output.to(torch.float16)
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def npu_op_exec(self, input1):
        input1.requires_grad = True
        output = torch.nn.functional.elu(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output = output.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.detach().numpy()
        return output, output_grad

    def test_elu_common_shape_format_fp16(self):
        shape_format = [
            [[np.float16, 0, (65535, 1, 1, 1)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 8192)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 16384)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 32768)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 65535)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 131072)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 196608)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 262144)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 393216)], -2, 2],
            [[np.float16, 2, (1, 1, 1, 524288)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 655360)], -2, 2],
            [[np.float16, 0, (1, 1, 1, 786432)], -2, 2],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1)
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)

    def test_elu_common_shape_format_fp32(self):
        shape_format1 = [
            [[np.float32, 0, (1, 31, 149, 2)], -1.1754943508e-38, -1.1754943508e-38],
            [[np.float32, 0, (1, 32, 31, 1)], -3402823500.0, 3402823500.0],
            [[np.float32, 0, (128,)], 3402823500, 3402800000],
            [[np.float32, 0, (184965, 1)], -9.313225746154785e-10, 9.313225746154785e-10],
            [[np.float32, 0, (1, 31, 149, 2)], -3402823500.0, -3402823500.0],
            [[np.float32, 2, (1, 31, 149, 2)], -3402823500.0, 3402823500.0],
            [[np.float32, 0, (1, 31, 149, 2)], -9.313225746154785e-10, 9.313225746154785e-10],
            [[np.float32, 0, (2, 31, 149, 2)], -0.000000000000000000000000000000000000011754943508,
             0.000000000000000000000000000000000000011754943508],
            [[np.float32, 0, (4, 31, 149, 2)], 0.000000000000000000000000000000000000011754943508,
             0.000000000000000000000000000000000000011754943508],
            [[np.float32, 0, (2048, 31, 1, 2)], -0.000000000000000000000000000000000000011754943508,
             -0.000000000000000000000000000000000000011754943508],
            [[np.float32, 0, (8, 7, 149)], -0.000000000000000000000000000000000000011754943508,
             0.000000000000000000000000000000000000011754943508]
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            cpu_output, cpu_output_grad = self.cpu_op_exec(cpu_input1)
            npu_output, npu_output_grad = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_grad, npu_output_grad)


if __name__ == '__main__':
    run_tests()
