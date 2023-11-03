import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestFillDiagonal(TestCase):
    def npu_op_exec(self, input1):
        input1 = input1.npu()
        input1.fill_diagonal_(1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec(self, input1):
        input1.fill_diagonal_(1)
        output = input1.numpy()
        return output

    def npu_op_wrap_exec(self, input1):
        input1 = input1.npu()
        input1.fill_diagonal_(1, wrap=True)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_wrap_exec(self, input1):
        input1.fill_diagonal_(1, wrap=True)
        output = input1.numpy()
        return output

    def npu_op_wrap_false_exec(self, input1):
        input1 = input1.npu()
        input1.fill_diagonal_(1, wrap=False)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_wrap_false_exec(self, input1):
        input1.fill_diagonal_(1, wrap=False)
        output = input1.numpy()
        return output

    def test_fill_diagonal_shape_format_fp32(self, device="npu"):
        format_list = [0, 3]
        shape_list = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item1 in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item1, 0, 100)
            cpu_input2 = cpu_input1.clone()
            npu_input2 = npu_input1.clone()
            cpu_output2 = self.cpu_op_exec(cpu_input1)
            npu_output2 = self.npu_op_exec(npu_input1)
            cpu_output3 = self.cpu_op_wrap_exec(cpu_input2)
            npu_output3 = self.npu_op_wrap_exec(npu_input2)
            self.assertRtolEqual(cpu_output2, npu_output2)
            self.assertRtolEqual(cpu_output3, npu_output3)

    def test_fill_diagonal_shape_format_fp16(self, device="npu"):
        format_list = [0, 3]
        shape_list = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input1 = cpu_input.clone()
            npu_input1 = npu_input.clone()
            cpu_output1 = self.cpu_op_exec(cpu_input)
            npu_output1 = self.npu_op_exec(npu_input)
            cpu_output2 = self.cpu_op_wrap_exec(cpu_input1)
            npu_output2 = self.npu_op_wrap_exec(npu_input1)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_fill_diagonal_false_shape_format_fp32(self, device="npu"):
        format_list1 = [0, 3]
        shape_list1 = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float32, i, j] for i in format_list1 for j in shape_list1
        ]
        for item1 in shape_format:
            cpu_input2, npu_input2 = create_common_tensor(item1, 0, 100)
            cpu_input3 = cpu_input2.clone()
            npu_input3 = npu_input2.clone()
            cpu_output3 = self.cpu_op_exec(cpu_input2)
            npu_output3 = self.npu_op_exec(npu_input2)
            cpu_output4 = self.cpu_op_wrap_false_exec(cpu_input3)
            npu_output4 = self.npu_op_wrap_false_exec(npu_input3)
            self.assertRtolEqual(cpu_output3, npu_output3)
            self.assertRtolEqual(cpu_output4, npu_output4)

    def test_fill_diagonal_false_shape_format_fp16(self, device="npu"):
        format_list1 = [0, 3]
        shape_list1 = ([7, 3], [3, 3, 3])
        shape_format = [
            [np.float16, i, j] for i in format_list1 for j in shape_list1
        ]
        for item in shape_format:
            cpu_input3, npu_input3 = create_common_tensor(item, 0, 100)
            cpu_input4 = cpu_input3.clone()
            npu_input4 = npu_input3.clone()
            cpu_output4 = self.cpu_op_exec(cpu_input3)
            npu_output4 = self.npu_op_exec(npu_input3)
            cpu_output5 = self.cpu_op_wrap_false_exec(cpu_input4)
            npu_output5 = self.npu_op_wrap_false_exec(npu_input4)
            self.assertRtolEqual(cpu_output4, npu_output4)
            self.assertRtolEqual(cpu_output5, npu_output5)


if __name__ == "__main__":
    run_tests()
