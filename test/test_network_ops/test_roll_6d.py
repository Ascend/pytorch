import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestRoll6d(TestCase):
    def generate_data(self, min_d1, max_d1, shape1, dtype1):
        input1 = np.random.uniform(min_d1, max_d1, shape1).astype(dtype1)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input1, shifts1, dims1):
        output1 = torch.roll(input1, shifts1, dims1).numpy()
        return output1

    def npu_op_exec(self, input1, shifts1, dims1):
        input2 = input1.to("npu")
        output1 = torch.roll(input2, shifts1, dims1)
        output1 = output1.to("cpu")
        output1 = output1.numpy()
        return output1

    def test_roll_10_10_10_10_10_10_int8(self):
        input1 = self.generate_data(-1, 1, (10, 10, 10, 10, 10, 10), np.int8)
        cpu_output1 = self.cpu_op_exec(input1, [-20, 30, 5], [-3, -4, -5])
        npu_output1 = self.npu_op_exec(input1, [-20, 30, 5], [-3, -4, -5])
        self.assertRtolEqual(cpu_output1, npu_output1)


if __name__ == "__main__":
    run_tests()
