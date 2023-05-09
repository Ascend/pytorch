import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestTriuIndices(TestCase):

    def cpu_op_exec(self, r, c):
        output = torch.triu_indices(r, c, device="cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, r, c):
        output = torch.triu_indices(r, c, device="npu")
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_tril_indices(self):
        shape_format = [
            [3, 3],
            [10, 10],
            [100, 100]
        ]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1])
            npu_output = self.npu_op_exec(item[0], item[1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
