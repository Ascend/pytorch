import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestRound(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.round(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.round(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_(self, input1):
        output = torch.round_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec_(self, input1):
        output = torch.round_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, cpu_out):
        output = torch.round(input1, out=cpu_out)
        output = cpu_out.numpy()
        return output

    def npu_op_exec_out(self, input1, npu_out):
        output = torch.round(input1, out=npu_out)
        output = npu_out.to("cpu")
        output = output.numpy()
        return output

    def test_round_float32_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (3)]],
            [[np.float32, -1, (4, 23)]],
            [[np.float32, -1, (2, 3)]],
            [[np.float32, -1, (12, 23)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_round_inp_float32_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (14)]],
            [[np.float32, -1, (4, 3)]],
            [[np.float32, -1, (12, 32)]],
            [[np.float32, -1, (22, 38)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec_(cpu_input1)
            npu_output = self.npu_op_exec_(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_round_out_common_shape_format(self):
        shape_format = [
            [[np.float16, -1, (10, 5)], [np.float16, -1, (5, 2)]],
            [[np.float16, -1, (4, 1, 5)], [np.float16, -1, (8, 1, 10)]],
            [[np.float32, -1, (10)], [np.float32, -1, (5)]],
            [[np.float32, -1, (4, 1, 5)], [np.float32, -1, (8, 1, 3)]],
            [[np.float32, -1, (2, 3, 8)], [np.float32, -1, (2, 3, 16)]],
            [[np.float32, -1, (2, 13, 56)], [np.float32, -1, (1, 26, 56)]],
            [[np.float32, -1, (2, 13, 56)], [np.float32, -1, (1, 26)]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            cpu_out1, npu_out1 = create_common_tensor(item[0], 1, 100)
            cpu_out2, npu_out2 = create_common_tensor(item[1], 1, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            if cpu_out1.dtype == torch.float16:
                cpu_out1 = cpu_out1.to(torch.float32)
            cpu_output = self.cpu_op_exec_out(cpu_input1, cpu_out1)
            npu_output1 = self.npu_op_exec_out(npu_input1, npu_out1)
            npu_output2 = self.npu_op_exec_out(npu_input1, npu_out2)
            cpu_output = cpu_output.astype(npu_output1.dtype)
            self.assertRtolEqual(cpu_output, npu_output1)
            self.assertRtolEqual(cpu_output, npu_output2)


if __name__ == "__main__":
    run_tests()
