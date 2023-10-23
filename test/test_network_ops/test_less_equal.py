import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestLessEqual(TestCase):
    def generate_scalar(self, min1, max1):
        scalar = np.random.uniform(min1, max1)
        return scalar

    def cpu_op_exec(self, input1, input2):
        output = torch.less_equal(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, input3):
        torch.less_equal(input1, input2, out=input3)
        output = input3.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.less_equal(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec(self, input1, input2):
        output = input1.less_equal_(input2)
        output = input1
        output = output.numpy()
        return output

    def npu_op_inplace_exec(self, input1, input2):
        input1.less_equal_(input2)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, output):
        torch.less_equal(input1, input2, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_scalar(self, input1, scalar):
        output = torch.less_equal(input1, scalar)
        output = output.numpy()
        return output

    def cpu_op_exec_scalar_out(self, input1, scalar, input2):
        torch.less_equal(input1, scalar, out=input2)
        output = input2.numpy()
        return output

    def npu_op_exec_scalar(self, input1, scalar):
        output = torch.less_equal(input1, scalar)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_inplace_exec_scalar(self, input1, scalar):
        input1.less_equal_(scalar)
        output = input1.numpy()
        return output

    def npu_op_inplace_exec_scalar(self, input1, scalar):
        input1.less_equal_(scalar)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_scalar_out(self, input1, scalar, output):
        torch.less_equal(input1, scalar, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def less_equal_tensor_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3 = torch.randn(item[1][2]) < 0
            npu_input3 = cpu_input3.npu()
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            if cpu_input3.dtype == torch.float16:
                cpu_input3 = cpu_input3.to(torch.float32)
            cpu_output_out = self.cpu_op_exec_out(cpu_input1, cpu_input2, cpu_input3)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_input3)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_less_equal_tensor_out(self):
        shape_format = [
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [128, 3, 224, 224]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [128, 116, 14, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [256, 128, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 0, [2, 3, 3, 3]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [128, 232, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.less_equal_tensor_out_result(shape_format)

    def less_equal_scalar_out_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2 = torch.randn(item[1][2]) < 0
            npu_input2 = cpu_input2.npu()
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_input2.dtype == torch.float16:
                cpu_input2 = cpu_input2.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output_out = self.cpu_op_exec_scalar_out(cpu_input1, scalar, cpu_input2)
            npu_output_out = self.npu_op_exec_scalar_out(npu_input1, scalar, npu_input2)
            cpu_output_out = cpu_output_out.astype(npu_output_out.dtype)
            self.assertRtolEqual(cpu_output_out, npu_output_out)

    def test_less_equal_scalar_out(self):
        shape_format = [
            [[np.float16, 0, [12, 4, 12, 121]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 0, [12, 10, 14, 111]], [np.float16, 0, [256, 116, 1, 1]]],
            [[np.float16, 2, [16, 3, 11, 121, 21]], [np.float16, 0, [3, 3, 3]]],
            [[np.float16, 0, [16, 16, 14]], [np.float16, 0, [128, 116, 14, 14]]],
            [[np.float32, 0, [20, 10, 7, 7]], [np.float32, 0, [128, 256, 3, 3]]],
            [[np.float32, 2, [1313, 3, 3, 3, 121]], [np.float32, 0, [3, 1, 3]]],
            [[np.float32, 0, [16, 22, 7, 7]], [np.float32, 0, [128, 232, 7, 7]]],
        ]
        self.less_equal_scalar_out_result(shape_format)

    def test_less_equal_scalar_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_scalar_int32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.int32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_scalar_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_exec_scalar(npu_input, scalar)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_tensor_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_tensor_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float16, i, j], [np.float16, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_inplace_float32(self):
        format_list = [0, 3]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float32, i, j], [np.float32, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_inplace_float16(self):
        format_list = [0, 3]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [[[np.float16, i, j], [np.float16, i, j]]
                        for i in format_list for j in shape_list]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_input1 = cpu_input1.to(torch.float32)
            cpu_input2 = cpu_input2.to(torch.float32)
            cpu_output = self.cpu_op_inplace_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_inplace_exec(npu_input1, npu_input2)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_inplace_scalar_float32(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            scalar = self.generate_scalar(0, 100)
            scalar1 = copy.deepcopy(scalar)
            npu_input = copy.deepcopy(cpu_input)
            cpu_output = self.cpu_op_inplace_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_inplace_exec_scalar(npu_input, scalar1)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_inplace_scalar_float16(self):
        format_list = [0]
        shape_list = [(5, 3), (2, 3, 4)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            scalar = self.generate_scalar(0, 100)
            cpu_output = self.cpu_op_inplace_exec_scalar(cpu_input, scalar)
            npu_output = self.npu_op_inplace_exec_scalar(npu_input, scalar)
            cpu_output = cpu_output.astype(np.float16)
            self.assertEqual(cpu_output, npu_output)

    def test_less_equal_mix_dtype(self):
        npu_input1, npu_input2 = create_common_tensor([np.float16, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3)
        npu_output = self.npu_op_exec(npu_input2, npu_input4)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
