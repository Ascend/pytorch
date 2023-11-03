import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestZerosLike(TestCase):
    def cpu_op_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_zeroslike_fp32(self):
        format_list = [0, 3, 29]
        shape_list = [1, (1000, 1280), (32, 3, 3), (32, 144, 1, 1)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, torch.float32)
            npu_output = self.npu_op_exec(npu_input, torch.float32)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_zeroslike_fp16(self):
        format_list = [0, 3, 29]
        shape_list = [1, (1000, 1280), (32, 3, 3), (32, 144, 1, 1)]
        shape_format = [
            [np.float16, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_input = cpu_input.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input, torch.float16)
            npu_output = self.npu_op_exec(npu_input, torch.float16)
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
