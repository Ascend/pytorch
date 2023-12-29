import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


class TestGridSampler3D(TestCase):
    def exec_grid_sampler3d_fp32(self, interpolation_mode, padding_mode, align_corners):
        format_list = [2]
        shape_list = [[2, 100, 1, 28, 28], [2, 100, 64, 32, 28]]
        shape_format = [
            [np.float32, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float32, 2, [2, 100, 1, 1, 3]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 100)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = self.op_exec_com(0,
                                          cpu_input, cpu_sample, interpolation_mode, padding_mode, align_corners)
            npu_output = self.op_exec_com(1,
                                          npu_input, npu_sample, interpolation_mode, padding_mode, align_corners)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_grid_sampler3d_fp32(self, device="npu"):
        self.exec_grid_sampler3d_fp32(0, 0, True)
        self.exec_grid_sampler3d_fp32(0, 1, True)
        self.exec_grid_sampler3d_fp32(1, 0, True)
        self.exec_grid_sampler3d_fp32(1, 1, True)
        self.exec_grid_sampler3d_fp32(0, 0, False)
        self.exec_grid_sampler3d_fp32(0, 1, False)
        self.exec_grid_sampler3d_fp32(1, 0, False)
        self.exec_grid_sampler3d_fp32(1, 1, False)

    def test_grid_sampler3d_fp16(self, device="npu"):
        format_list = [2]
        shape_list = [[2, 1, 1, 3, 3], [2, 1, 2, 3, 4]]
        shape_format = [
            [np.float16, j, k] for j in format_list for k in shape_list
        ]
        sample_format = [np.float16, 2, [2, 1, 2, 2, 3]]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 10)
            cpu_sample, npu_sample = create_common_tensor(sample_format, -1, 1)
            cpu_output = self.cpu_op_fp16_exec(cpu_input, cpu_sample, 0, 0, True)
            npu_output = self.op_exec_com(1, npu_input, npu_sample, 0, 0, True)
            self.assertRtolEqual(cpu_output, npu_output)

    def op_exec_com(self, npu_flag, input1, sample, interpolation_mode, padding_mode, align_corners):
        output = torch.grid_sampler_3d(input1, sample, interpolation_mode, padding_mode, align_corners)
        if npu_flag:
            return output.to("cpu").numpy()
        return output.numpy()

    def cpu_op_fp16_exec(self, input1, sample, interpolation_mode, padding_mode, align_corners):
        input1 = input1.to(torch.float32)
        sample = sample.to(torch.float32)
        output = torch.grid_sampler_3d(input1, sample, interpolation_mode, padding_mode, align_corners)
        output = output.numpy().astype(np.float16)
        return output


if __name__ == "__main__":
    np.random.seed(1234)
    run_tests()
