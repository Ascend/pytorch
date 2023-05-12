import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestAcosh(TestCase):

    def cpu_op_exec(self, input1):
        output = torch.acosh(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch.acosh(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        torch.acosh(input1, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def cpu_inp_op_exec(self, input1):
        output = torch.acosh_(input1)
        output = output.numpy()
        return output

    def npu_inp_op_exec(self, input1):
        output = torch.acosh_(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_acosh_common_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, (5, 3)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
            self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_acosh_out_common_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, (4, 3)], [np.float32, 0, (4, 3)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item[1], -10, 10)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec_out(npu_input1, npu_input2)
            mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
            self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)

    def test_acosh_inp_common_shape_format(self):
        shape_format1 = [
            [[np.float32, 0, (5, 3)]],
        ]
        for item in shape_format1:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -10, 10)
            cpu_output = self.cpu_inp_op_exec(cpu_input1)
            npu_output = self.npu_inp_op_exec(npu_input1)
            mask = ~(np.isnan(cpu_output) | np.isinf(cpu_output))
            self.assertRtolEqual(cpu_output[mask], npu_output[mask], 0.001)


if __name__ == "__main__":
    run_tests()