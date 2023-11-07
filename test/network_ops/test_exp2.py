import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestExp2(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.exp2(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.exp2(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        input1.exp2_()
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, out):
        torch.exp2(input1, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_exp2_shape_format_fp16(self):
        format_list = [0, 2, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            out = npu_input
            npu_output1 = self.npu_op_exec_out(npu_input, out)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output1)

    def test_exp2_shape_format_fp32(self):
        format_list = [0, 2, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            out = npu_input
            npu_output1 = self.npu_op_exec_out(npu_input, out)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output1)

    def test_exp2_inp_shape_format_fp16(self):
        format_list = [0, 2, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_inp_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp2_inp_shape_format_fp32(self):
        format_list = [0, 2, 3]
        shape_list = [[5], [2, 4], [2, 2, 4], [2, 3, 3, 4]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_inp_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp2_inp_uncon_shape_format_fp32(self):
        format_list = [0, 2, 3]
        shape_list = [[8, 6]]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input1 = cpu_input1.transpose(0, 1)
            npu_input1 = npu_input1.transpose(0, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_exp2_inp_uncon_shape_format_fp16(self):
        format_list = [0, 2, 3]
        shape_list = [[8, 6]]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -1, 1)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input1 = cpu_input1.transpose(0, 1)
            npu_input1 = npu_input1.transpose(0, 1)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
