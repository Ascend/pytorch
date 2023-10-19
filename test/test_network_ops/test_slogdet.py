import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSlogdet(TestCase):
    def cpu_op_exec(self, input1):
        sign, logabsdet = torch.slogdet(input1)
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def npu_op_exec(self, input1):
        sign, logabsdet = torch.slogdet(input1)
        sign = sign.cpu()
        logabsdet = logabsdet.cpu()
        sign = sign.numpy()
        logabsdet = logabsdet.numpy()
        return sign, logabsdet

    def test_slogdet_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, -1, (3, 3)],
            [np.float32, -1, (4, 3, 3)],
            [np.float32, -1, (5, 5, 5, 5)],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, -100, 100)
            cpu_output, cpu_indices = self.cpu_op_exec(cpu_input)
            npu_output, npu_indices = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_indices, npu_indices)


if __name__ == "__main__":
    run_tests()
