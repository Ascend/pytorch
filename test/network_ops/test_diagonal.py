import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDiagonal(TestCase):
    def cpu_tensor_op_exec(self, input1, offset):
        output = input1.diagonal(offset)
        output = output.numpy()
        return output

    def npu_tensor_op_exec(self, input1, offset):
        output = input1.diagonal(offset)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def generate_npu_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        output1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_output1 = torch.from_numpy(output1)
        return npu_input1, npu_output1

    def test_diagonal_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, [10, 15]], 3],
            [[np.float32, -1, [6, 7, 8]], -4],
            [[np.float32, -1, [2, 4, 6]], 0],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_tensor_op_exec(cpu_input, item[1])
            npu_output = self.npu_tensor_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_diagonal_float16_shape_format(self):
        shape_format = [
            [[np.float16, -1, [10, 15]], 3],
            [[np.float16, -1, [6, 7, 8]], -4],
            [[np.float16, -1, [2, 4, 6]], 0],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_tensor_op_exec(cpu_input, item[1])
            npu_output = self.npu_tensor_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
