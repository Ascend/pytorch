import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestTransepose(TestCase):
    def test_transepose(self, device="npu"):
        def cpu_op_exec(input1, perm):
            output = input1.permute(perm)
            output = output.numpy()
            return output

        def npu_op_exec(input1, perm):
            output = torch_npu.npu_transpose(input1, perm)
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
            [[np.float32, 0, (5, 3, 6, 4)], [1, 0, 2, 3]],
            [[np.float16, 0, (5, 3, 6, 4)], [0, 3, 2, 1]],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = cpu_op_exec(cpu_input1, item[1])
            npu_output = npu_op_exec(npu_input1, item[1])

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
