import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLog2(TestCase):
    def cpu_op_exec(self, input1):
        output = torch.log2(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.log2(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1):
        output = input1
        torch.log2(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.log2_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        output = torch.log2_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log2_(input1)
        output = output.numpy()
        return output

    def npu_inp_uncon_op_exec(self, input1):
        input1 = input1.as_strided([2, 2], [1, 2], 2)
        output = torch.log2_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_log2_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log2_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log2_inp_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [(5, 3)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log2_inp_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log2_inp_uncon_shape_format_fp32(self):
        format_list = [0, 3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_log_inp_uncon_shape_format_fp16(self):
        format_list = [0, 3]
        shape_list = [(8, 6)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_inp_uncon_op_exec(cpu_input1)
            npu_output = self.npu_inp_uncon_op_exec(npu_input1)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    np.random.seed(1234)
    run_tests()
