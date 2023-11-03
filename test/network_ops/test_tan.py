import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestTan(TestCase):

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)

        return npu_input1

    def cpu_op_exec(self, input1):
        output = torch.tan(input1)
        output = output.numpy()
        return output

    def cpu_op_exec_out(self, input1, input2):
        torch.tan(input1, out=input2)
        output = input2.numpy()
        return output

    def cpu_op_exec_self(self, input1):
        torch.tan_(input1)
        output = input1.numpy()
        return output

    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch.tan(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, input2):
        input1 = input1.to("npu")
        output = input2.to("npu")
        torch.tan(input1, out=output)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_self(self, input1):
        input1 = input1.to("npu")
        torch.tan_(input1)
        output = input1.to("cpu")
        output = output.numpy()
        return output

    def test_tan_float32(self):
        input1 = self.generate_single_data(0, 6, (1, 3), np.float32)
        cpu_output = self.cpu_op_exec(input1)
        npu_output = self.npu_op_exec(input1)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tan_out_float32(self):
        input1 = self.generate_single_data(0, 6, (1, 3), np.float32)
        input2 = self.generate_single_data(0, 6, (1, 3), np.float32)
        cpu_output = self.cpu_op_exec_out(input1, input2)
        npu_output = self.npu_op_exec_out(input1, input2)
        self.assertRtolEqual(cpu_output, npu_output)

    def test_tan_self_float32(self):
        input1 = self.generate_single_data(0, 6, (1, 3), np.float32)
        input2 = copy.deepcopy(input1)
        cpu_output = self.cpu_op_exec_self(input1)
        npu_output = self.npu_op_exec_self(input2)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
