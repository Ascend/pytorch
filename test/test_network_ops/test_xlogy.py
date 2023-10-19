import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestXlogy(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = torch.xlogy(input1, input2)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.xlogy(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, output):
        torch.xlogy(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1, input2):
        output = torch.xlogy_(input1, input2)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1, input2):
        output = torch.xlogy_(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_xlogy_shape_format_fp32(self):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            # xlogy.Tensor
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
            # xlogy.Scalar_input1
            cpu_output = self.cpu_op_exec(2, cpu_input2)
            npu_output = self.npu_op_exec(2, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
            # xlogy.Scalar_input2
            cpu_output = self.cpu_op_exec(cpu_input1, 4)
            npu_output = self.npu_op_exec(npu_input1, 4)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_xlogy_out_shape_format_fp32(self):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item, 0, 100)
            # xlogy.outTensor
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            # xlogy.outScalar_input1
            cpu_output = self.cpu_op_exec(3, cpu_input2)
            npu_output = self.npu_op_exec_out(3, npu_input2, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)
            # xlogy.outScalar_input2
            cpu_output = self.cpu_op_exec(cpu_input1, 5)
            npu_output = self.npu_op_exec_out(npu_input1, 5, npu_input3)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_xlogy_inp_shape_format_fp32(self):
        format_list = [3]
        shape_list = [(4, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item, 0, 100)
            # xlogy_.Tensor
            cpu_output = self.cpu_inp_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_inp_op_exec(npu_input1, npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)
            # xlogy_.Scalar
            cpu_output = self.cpu_inp_op_exec(cpu_input1, 6)
            npu_output = self.npu_inp_op_exec(npu_input1, 6)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
