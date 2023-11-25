import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class Test__And__(TestCase):
    def cpu_op_exec(self, input1, input2):
        output = input1.__and__(input2)
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        return output.numpy()

    def npu_op_exec(self, input1, input2):
        output = input1.__and__(input2)
        output = output.to("cpu")
        if output.dtype != torch.int32:
            output = output.to(torch.int32)
        return output.numpy()

    def test___And___shape_format(self, device="npu"):
        shape_format = [
            [[np.int32, 0, [256, 1000]], [1]],
            [[np.int32, 0, [256, 1000]], [np.int32, 0, [256, 1000]]],
            [[np.int16, 0, [256, 1000]], [2]],
            [[np.int16, 0, [256, 1000]], [np.int16, 0, [256, 1000]]],
            [[np.int8, 0, [256, 1000]], [3]],
            [[np.int8, 0, [256, 1000]], [np.int8, 0, [256, 1000]]],
            [[np.bool_, 0, [256, 1000]], [np.bool_, 0, [256, 1000]]],
        ]

        for item in shape_format:
            if len(item[1]) > 1:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                cpu_input2, npu_input2 = create_common_tensor(item[1], 0, 100)
                cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2)
                npu_output = self.npu_op_exec(npu_input1, npu_input2)
                self.assertRtolEqual(cpu_output, npu_output)
            else:
                cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
                cpu_output = self.cpu_op_exec(cpu_input1, item[1][0])
                npu_output = self.npu_op_exec(npu_input1, item[1][0])
                self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
