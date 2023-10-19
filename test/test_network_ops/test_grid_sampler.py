import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGridSampler(TestCase):
    def test_grid_sampler_fp32(self):
        format_list = [0]
        shape_list = [[100, 1, 28, 28], [100, 64, 32, 28]]
        shape_format = [
            [np.float32, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float32, 0, [100, 1, 1, 2]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = self.cpu_op_exec(cpu_input, cpu_sample)
            npu_output = self.npu_op_exec(npu_input, npu_sample)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_grid_sampler_fp16(self):
        format_list = [0]
        shape_list = [[1, 1, 3, 3], [1, 2, 3, 4]]
        shape_format = [
            [np.float16, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float16, 0, [1, 2, 2, 2]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 10)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = self.cpu_op_fp16_exec(cpu_input, cpu_sample)
            npu_output = self.npu_op_exec(npu_input, npu_sample)
            self.assertRtolEqual(cpu_output, npu_output)

    def cpu_op_exec(self, input1, sample):
        output = torch.grid_sampler(input1, sample, 0, 0, True)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, sample):
        output = torch.grid_sampler(input1, sample, 0, 0, True)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_op_fp16_exec(self, input1, sample):
        input1 = input1.to(torch.float32)
        sample = sample.to(torch.float32)
        output = torch.grid_sampler(input1, sample, 0, 0, True)
        output = output.numpy()
        output = output.astype(np.float16)
        return output


if __name__ == "__main__":
    run_tests()
