import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSearchsorted(TestCase):

    def cpu_sorted_input(self, input1):
        input_dim = input1.dim() - 1
        input_op, _ = input1.float().sort(input_dim)
        input_op = input_op.to(input1.dtype)
        return input_op

    def cpu_op_exec(self, input1, input2):
        output = torch.searchsorted(input1, input2)
        output = output.numpy()
        return output

    def cpu_op_exec_bool(self, input1, input2, out_int32, right):
        output = torch.searchsorted(input1, input2, out_int32=out_int32, right=right)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2):
        output = torch.searchsorted(input1, input2)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_bool(self, input1, input2, out_int32, right):
        output = torch.searchsorted(input1, input2, out_int32=out_int32, right=right)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2, out):
        torch.searchsorted(input1, input2, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def test_searchsorted_tensor_shape_format(self):
        shape_format = [
            [[np.int32, 0, [256, 40]], [np.int32, 0, [256, 20]]],
            [[np.int64, 0, [256, 40]], [np.int64, 0, [256, 20]]],
            [[np.float32, 0, [256, 40]], [np.float32, 0, [256, 20]]],
            [[np.int32, 0, [4, 12, 12, 128]], [np.int32, 0, [4, 12, 12, 23]]],
            [[np.int64, 0, [4, 12, 12, 128]], [np.int64, 0, [4, 12, 12, 23]]],
            [[np.float32, 0, [4, 12, 12, 128]], [np.float32, 0, [4, 12, 12, 23]]],
        ]

        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            _, npu_out = create_common_tensor(item[1], -10, 10)
            cpu_input1 = self.cpu_sorted_input(cpu_input1)
            npu_input1 = cpu_input1.npu()
            npu_out = npu_out.long()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
            npu_output = self.npu_op_exec(npu_input1, npu_input2)
            npu_output_out = self.npu_op_exec_out(npu_input1, npu_input2, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_searchsorted_tensor_bool(self):
        shape_format = [
            [[np.int32, 0, [256, 50]], [np.int32, 0, [256, 20]]],
            [[np.int64, 0, [256, 50]], [np.int64, 0, [256, 20]]],
            [[np.float32, 0, [256, 50]], [np.float32, 0, [256, 20]]],
        ]

        for item in shape_format:
            cpu_input1, _ = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_input1 = self.cpu_sorted_input(cpu_input1)
            npu_input1 = cpu_input1.npu()
            cpu_output1 = self.cpu_op_exec_bool(cpu_input1, cpu_input2, True, False)
            npu_output1 = self.npu_op_exec_bool(npu_input1, npu_input2, True, False)
            cpu_output2 = self.cpu_op_exec_bool(cpu_input1, cpu_input2, False, True)
            npu_output2 = self.npu_op_exec_bool(npu_input1, npu_input2, False, True)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)

    def test_searchsorted_scalar_shape_format(self):
        shape_format = [
            [[np.int32, 0, [128]], 2],
            [[np.int64, 0, [256]], 3],
            [[np.float32, 0, [64]], 2.5],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -10, 10)
            cpu_input = self.cpu_sorted_input(cpu_input)
            npu_input = cpu_input.npu()
            scalar = item[1]
            cpu_output = self.cpu_op_exec(cpu_input, scalar)
            npu_output = self.npu_op_exec(npu_input, scalar)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_searchsorted_scalar_bool(self):
        shape_format = [
            [[np.float32, 0, [64]], 2.5],
            [[np.int32, 0, [128]], 2],
            [[np.int64, 0, [256]], 3]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -10, 10)
            cpu_input = self.cpu_sorted_input(cpu_input)
            npu_input = cpu_input.npu()
            scalar = item[1]
            cpu_output1 = self.cpu_op_exec_bool(cpu_input, scalar, True, False)
            npu_output1 = self.npu_op_exec_bool(npu_input, scalar, True, False)
            cpu_output2 = self.cpu_op_exec_bool(cpu_input, scalar, False, True)
            npu_output2 = self.npu_op_exec_bool(npu_input, scalar, False, True)
            self.assertRtolEqual(cpu_output1, npu_output1)
            self.assertRtolEqual(cpu_output2, npu_output2)


if __name__ == "__main__":
    run_tests()
