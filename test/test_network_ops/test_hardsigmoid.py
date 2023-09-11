import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestHardsigmoid(TestCase):

    def cpu_op_exec(self, input1):
        h = torch.nn.Hardsigmoid()
        output = h(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        h = torch.nn.Hardsigmoid()
        output = h(input1)
        output = output.cpu().numpy()
        return output

    def npu_op_exec_inp(self, input1):
        torch.nn.functional.hardsigmoid(input1, True)
        output = input1.cpu().numpy()
        return output

    def test_hardsigmoid(self):
        shape_foramt = [
            [np.int32, 0, (3, 6)],
            [np.float32, 0, (9, 3)],
            [np.float16, 0, (2, 7)]
        ]
        for item in shape_foramt:
            cpu_input, npu_input = create_common_tensor(item, -6, 6)
            if item[0] != np.float32:
                cpu_output = self.cpu_op_exec(cpu_input.float()).astype(item[0])
            else:
                cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            npu_inp_output = self.npu_op_exec_inp(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_inp_output)


if __name__ == '__main__':
    run_tests()
