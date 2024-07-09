import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestOne_(TestCase):
    def custom_op_exec(self, input1):
        output = torch.ones_like(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1):
        output = torch_npu.one_(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_one_(self):
        shape_format = [
            [np.float32, 0, (2, 3)],
            [np.float32, 0, (2, 3, 4)]
        ]
        for item in shape_format:
            _, npu_input1 = create_common_tensor(item, 0, 100)
            custom_output = self.custom_op_exec(npu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
