import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSqrt(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.sqrt(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.sqrt(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, input1, output):
        torch.sqrt(input1, out=output)
        output1 = output.numpy()
        return output1

    def npu_op_out_exec(self, input1, output):
        torch.sqrt(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        torch.sqrt_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        torch.sqrt_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_sqrt_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 6, 4)]],
            [[np.float32, 3, (2, 4, 5)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sqrt_shape_format_fp16(self):

        def cpu_op_exec_fp16(input1):
            input1 = input1.to(torch.float32)
            output = torch.sqrt(input1)
            output = output.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 0, (1, 6, 4)]],
            [[np.float16, 0, (2, 4, 5)]]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sqrt_out_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 6, 4)], [np.float32, 0, (1, 6, 4)]],
            [[np.float32, 3, (2, 4, 5)], [np.float32, 3, (2, 4, 5)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_out, npu_out = create_common_tensor(item[1], 1, 100)
            cpu_output = self.cpu_op_out_exec(cpu_input, cpu_out)
            npu_output = self.npu_op_out_exec(npu_input, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sqrt_out_shape_format_fp16(self):

        def cpu_op_out_exec_fp16(input1, output):
            input1 = input1.to(torch.float32)
            output = output.to(torch.float32)
            torch.sqrt(input1, out=output)
            output1 = output.numpy()
            output1 = output1.astype(np.float16)
            return output1

        shape_format = [
            [[np.float16, 0, (1, 6, 4)], [np.float16, 0, (1, 6, 4)]],
            [[np.float16, 0, (2, 4, 5)], [np.float16, 0, (2, 4, 5)]]
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_out, npu_out = create_common_tensor(item[1], 1, 100)
            cpu_output = cpu_op_out_exec_fp16(cpu_input, cpu_out)
            npu_output = self.npu_op_out_exec(npu_input, npu_out)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sqrt1_shape_format(self):
        shape_format = [
            [[np.float32, 0, (1, 6, 4)]],
            [[np.float32, 3, (2, 4, 5)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input)
            npu_output = self.npu_op_exec_(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_sqrt1_shape_format_fp16(self):
        def cpu_op_exec_fp16_(input1):
            input1 = input1.to(torch.float32)
            torch.sqrt_(input1)
            output = input1.numpy()
            output = output.astype(np.float16)
            return output

        shape_format = [
            [[np.float16, 0, (1, 6, 4)]],
            [[np.float16, 0, (2, 4, 5)]]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_output = cpu_op_exec_fp16_(cpu_input)
            npu_output = self.npu_op_exec_(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
