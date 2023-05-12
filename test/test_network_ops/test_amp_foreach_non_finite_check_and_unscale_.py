import copy
import sys

import torch
import numpy as np

import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class Test_AmpForeachNonFiniteCheckAndUnscale_(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype, input3):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input1 = torch.from_numpy(input1)
        input2 = np.array([0.0]).astype(dtype)
        input2 = torch.from_numpy(input2)
        input3 = np.array([input3]).astype(dtype)
        input3 = torch.from_numpy(input3)
        return input1, input2, input3

    def cpu_op_exec(self, input1, input2, input3):
        input1 = input1.numpy()
        input2 = input2.numpy()
        input3 = input3.numpy()
        res = np.multiply(input1, input3)
        return res

    def npu_op_exec(self, input1, input2, input3):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        input3 = input3.to("npu")
        torch._amp_foreach_non_finite_check_and_unscale_((input1,),input2,input3)
        input1 = input1.to("cpu")
        input1 = input1.numpy()
        return input1


    def test_AmpForeachNonFiniteCheckAndUnscale_float32_case1(self, device='npu'):
        params = [(0, 100, (4, 3), np.float32, 1.5),
                  (0, 100, (2, 5, 6), np.float32, 3.7),
                  (0, 100, (5, 7), np.float32, 1.9),
                  (0, 100, (2, 8, 1), np.float32, 3.2)
                  ]
        for param in params:
            input1, input2, input3 = self.generate_data(*param)
            cpu_output = self.cpu_op_exec(input1, input2, input3)
            npu_output = self.npu_op_exec(input1, input2, input3)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
