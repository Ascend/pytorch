import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSub(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1 - input2
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1 - input2
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_t(self, input1, input2):
        output = torch.subtract(input1, input2, alpha=1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_t_out(self, input1, input2, input3):
        torch.subtract(input1, input2, alpha=1, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_t_out(self, input1, input2, input3):
        torch.subtract(input1, input2, alpha=1, out=input3)
        output = input3.numpy()
        return output

    def cpu_op_exec(self, input1, input2):
        output = input1.subtract(input2, alpha=1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = input1.subtract(input2, alpha=1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_inp(self, input1, input2):
        input1.subtract_(input2, alpha=1)
        output = input1.numpy()
        return output

    def npu_op_exec_inp(self, input1, input2):
        input1.subtract_(input2, alpha=1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def subtract_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[0], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input3.dtype == torch.float16:
                cpu_input3 = cpu_input3.to(torch.float32)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_t_out = self.cpu_op_exec_t_out(cpu_input1, cpu_input2, cpu_input3)
            npu_output_t_out = self.npu_op_exec_t_out(npu_input1, npu_input2, npu_input3)
            cpu_output_t_out = cpu_output_t_out.astype(npu_output_t_out.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_t_out, npu_output_t_out)
            cpu_input1_tensor, npu_input1_tensor = create_common_tensor(item[0], 0, 100)
            if cpu_input1_tensor.dtype == torch.float16:
                cpu_input1_tensor = cpu_input1_tensor.to(torch.float32)
            cpu_input2_tensor, npu_input2_tensor = create_common_tensor(item[0], 0, 100)
            if cpu_input2_tensor.dtype == torch.float16:
                cpu_input2_tensor = cpu_input2_tensor.to(torch.float32)
            cpu_output_inp_tensor = self.cpu_op_exec_inp(cpu_input1_tensor, cpu_input2_tensor)
            npu_output_inp_tensor = self.npu_op_exec_inp(npu_input1_tensor, npu_input2_tensor)
            cpu_output_inp_tensor = cpu_output_inp_tensor.astype(npu_output_inp_tensor.dtype)
            self.assertRtolEqual(cpu_output_inp_tensor, npu_output_inp_tensor)

    def test_subtract_scalar_shape_format_fp32_1d(self):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [448]], np.random.uniform(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_fp32_2d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [1000, 1280]], np.random.uniform(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_fp32_3d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3]], np.random.uniform(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_fp32_4d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [256, 480, 14, 14]], np.random.uniform(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_int32_1d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [448]], np.random.randint(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_int32_2d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7]], np.random.randint(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_int32_3d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7, 58]], np.random.randint(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar_shape_format_int32_4d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14, 14]], np.random.randint(0, 100)] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp16_1d(self):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float16, i, [448]], [np.float16, i, [448]]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp16_2d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [1000, 1280]], [np.float16, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp16_3d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3]], [np.float16, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp16_4d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float16, i, [256, 480, 14, 14]], [np.float16, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp32_1d(self):
        format_list = [-1, 0, 3]
        shape_format = [[[np.float32, i, [448]], [np.float32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp32_2d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [1000, 1280]], [np.float32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp32_3d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3]], [np.float32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_fp32_4d(self):
        format_list = [-1, 0, 3, 29]
        shape_format = [[[np.float32, i, [256, 480, 14, 14]], [np.float32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_int32_1d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [448]], [np.int32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_int32_2d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7]], [np.int32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_int32_3d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [64, 7, 58]], [np.int32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_shape_format_int32_4d(self):
        format_list = [-1, 0]
        shape_format = [[[np.int32, i, [256, 480, 14, 14]], [np.int32, i, []]] for i in format_list]
        self.subtract_result(shape_format)

    def test_subtract_scalar(self):
        format_list = [-1, 0]
        shape_list = [[448], [64, 7], [64, 7, 58], [256, 480, 14, 14]]
        dtype_list = [np.float32, np.float32, np.int32]
        shape_format = [
            [k, i, j] for k in dtype_list for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)
            scalar = 2
            cpu_output = self.cpu_op_exec(cpu_input, scalar)
            npu_output = self.npu_op_exec(npu_input, scalar)
            cpu_output_scalar = self.cpu_op_exec_inp(cpu_input, scalar)
            npu_output_scalar = self.npu_op_exec_inp(npu_input, scalar)
            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_output_scalar = cpu_output_scalar.astype(npu_output_scalar.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output_scalar, npu_output_scalar)


if __name__ == "__main__":
    run_tests()
