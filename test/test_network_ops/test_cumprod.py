import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestCumprod(TestCase):

    def cpu_op_exec(self, input1, dim):
        output = torch.cumprod(input1, dim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim):
        output = torch.cumprod(input1, dim)
        output = output.to("cpu").numpy()
        return output
    
    def npu_op_out_exec(self, input1, input2, dim):
        torch.cumprod(input1, dim, out=input2)
        output = input2.to("cpu").numpy()
        return output

    def cpu_op_exec_inp(self, input1, dim):
        input1.cumprod_(dim)
        return input1.numpy()

    def npu_op_exec_inp(self, input1, dim):
        input1.cumprod_(dim)
        return input1.cpu().numpy()

    def test_cumprod_common_shape_format(self):
        shape_format = [
            [[np.float32, 0, (5, 3)], 0],
            [[np.float32, 0, (2, 3)], 1]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            dim = item[1]
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_exec(npu_input1, dim)
            self.assertRtolEqual(cpu_output, npu_output)

            cpu_output_inp = self.cpu_op_exec_inp(cpu_input1, dim)
            npu_output_inp = self.npu_op_exec_inp(npu_input1, dim)
            self.assertRtolEqual(cpu_output_inp, npu_output_inp)

    def test_cumprod_out_common_shape_format(self):
        shape_format = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 4)]],
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 4, 1)]],
        ]
        dim = 0
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            _, npu_input2 = create_common_tensor(item[1], 1, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, dim)
            npu_output = self.npu_op_out_exec(npu_input1, npu_input2, dim)
            self.assertEqual(cpu_output.shape, npu_output.shape)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
