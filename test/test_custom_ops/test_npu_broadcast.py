import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuBroadcast(TestCase):
    def custom_op_exec(self, input1, shape):
        output = torch.broadcast_to(input1, shape)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, size):
        output = torch_npu.npu_broadcast(input1, size)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_npu_broadcast(self):
        input1 = [
            torch.tensor([1, 2, 3]).npu(),
            torch.tensor([[1], [2], [3]]).npu()
        ]
        for item in input1:
            custom_output = self.custom_op_exec(item, (3, 3))
            npu_output = self.npu_op_exec(item, (3, 3))
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
