import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestDiag(TestCase):
    def cpu_op_exec(self, input1, diagonal):
        output = torch.diag(input1, diagonal=diagonal)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, diagonal):
        output = torch.diag(input1, diagonal=diagonal)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, diagonal, out):
        torch.diag(input1, diagonal=diagonal, out=out)
        output = out.numpy()
        return output

    def npu_op_exec_out(self, input1, diagonal, out):
        torch.diag(input1, diagonal=diagonal, out=out)
        output = out.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_fp16(self, input1, diagonal):
        input1 = input1.to(torch.float32)
        output = torch.diag(input1, diagonal)
        output = output.numpy()
        output = output.astype(np.float16)
        return output

    def generate_npu_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        output1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        npu_output1 = torch.from_numpy(output1)
        return npu_input1, npu_output1

    def test_diag_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, [16]], 0],    # test the condition of 1-dimension
            [[np.float32, -1, [1024]], 0],
            [[np.float32, -1, [5, 5]], 0],  # test the condition of 2-dimension
            [[np.float32, -1, [256, 256]], 0],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_diag_float32_out(self):
        shape_format = [
            [[np.float32, -1, [16]], [np.float32, -1, [20]], 0],    # test the condition of 1-dimension
            [[np.float32, -1, [1024]], [np.float32, -1, [20, 20]], 0],
            [[np.float32, -1, [5, 5]], [np.float32, -1, [5, 5, 5]], 0],  # test the condition of 2-dimension
            [[np.float32, -1, [256, 256]], [np.float32, -1, [256]], 0],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input1, item[2])
            npu_output = self.npu_op_exec_out(npu_input1, item[2], npu_input2)
            self.assertRtolEqual(cpu_output, npu_output)

        npu_input1, npu_output1 = self.generate_npu_data(0, 100, (5, 5), np.float32)
        cpu_output = self.cpu_op_exec(npu_input1, 0)
        npu_output = self.npu_op_exec_out(npu_input1, 0, npu_output1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_diag_float16_out(self):
        shape_format = [
            [[np.float16, -1, [16]], [np.float16, -1, [20]], 0],    # test the condition of 1-dimension
            [[np.float16, -1, [1024]], [np.float16, -1, [20, 20]], 0],
            [[np.float16, -1, [5, 5]], [np.float16, -1, [5, 5, 5]], 0],  # test the condition of 2-dimension
            [[np.float16, -1, [256, 256]], [np.float16, -1, [256]], 0],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, item[2])
            npu_output = self.npu_op_exec_out(npu_input1, item[2], npu_input2)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_diag_float16_shape_format(self):
        shape_format = [
            [[np.float16, -1, [4]], 0],     # test the condition of 1-dimension
            [[np.float16, -1, [512]], 0],
            [[np.float16, -1, [4, 4]], 0],  # test the condition of 2-dimension
            [[np.float16, -1, [256, 256]], 0],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_fp16(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
