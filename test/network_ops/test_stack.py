import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestStack(TestCase):
    def cpu_op_exec(self, input1, input2, dim):
        cpu_output = torch.stack((input1, input2), dim)
        cpu_output = cpu_output.numpy()
        return cpu_output

    def npu_op_exec(self, input1, input2, dim):
        output = torch.stack((input1, input2), dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2, dim, input3):
        torch.stack((input1, input2), dim, out=input3)
        output = input3.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, dim, input3):
        torch.stack((input1, input2), dim, out=input3)
        output = input3.to("cpu")
        output = output.numpy()
        return output

    def npu_output_size(self, inputs, dim=0):
        shape = []
        for i in range(dim):
            shape.append(inputs[0].size(i))
        shape.append(len(inputs))
        for i in range(dim, inputs[0].dim()):
            shape.append(inputs[0].size(i))

        return shape

    def stack_result(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], 0, 100)
            shape = self.npu_output_size([npu_input1, npu_input2], item[1])
            npu_input3 = torch.ones(shape, dtype=cpu_input1.dtype).npu()
            cpu_input3 = torch.ones(shape, dtype=cpu_input1.dtype)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
                cpu_input2 = cpu_input2.to(torch.float32)
                cpu_input3 = cpu_input3.to(torch.float32)

            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, item[1])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[1])
            cpu_output_out = self.cpu_op_exec_out(cpu_input1, cpu_input2, item[1], cpu_input3)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, item[1], npu_input3)

            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_stack_shape_format_fp16_1d(self):
        format_list = [0, 3]
        shape_format = [[[np.float16, i, [18]], np.random.randint(0, 1)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp16_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [5, 256]], np.random.randint(0, 2)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp16_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [32, 3, 3]], np.random.randint(0, 3)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp16_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float16, i, [32, 32, 3, 3]], np.random.randint(0, 4)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp32_1d(self):
        format_list = [0, 3]
        shape_format = [[[np.float32, i, [18]], np.random.randint(0, 1)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp32_2d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [5, 256]], np.random.randint(0, 2)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp32_3d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [32, 3, 3]], np.random.randint(0, 3)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_fp32_4d(self):
        format_list = [0, 3, 29]
        shape_format = [[[np.float32, i, [32, 32, 3, 3]], np.random.randint(0, 4)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_int32_1d(self):
        format_list = [0]
        shape_format = [[[np.int32, i, [18]], np.random.randint(0, 1)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_int32_2d(self):
        format_list = [0]
        shape_format = [[[np.int32, i, [5, 256]], np.random.randint(0, 2)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_int32_3d(self):
        format_list = [0]
        shape_format = [[[np.int32, i, [32, 3, 3]], np.random.randint(0, 3)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_shape_format_int32_4d(self):
        format_list = [-1]
        shape_format = [[[np.int32, i, [32, 32, 3, 3]], np.random.randint(0, 4)] for i in format_list]
        self.stack_result(shape_format)

    def test_stack_size_dim(self):
        def cpu_op_exec(input1):
            output = torch.stack((input1, input1, input1, input1, input1, input1, input1, input1, input1))
            return output.numpy()

        def npu_op_exec(input1):
            output = torch.stack((input1, input1, input1, input1, input1, input1, input1, input1, input1))
            output = output.to("cpu")
            return output.numpy()
        shape_format = [
            [[np.int32, 0, ()]],
            [[np.float32, 0, ()]],
            [[np.float16, 0, ()]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec(cpu_input1)
            npu_output = npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
