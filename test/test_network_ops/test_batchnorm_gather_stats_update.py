import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNormGatherStatsUpdate(TestCase):
    def expect_cpu_out(self, cpu_sum, cpu_square_sum, cpu_running_mean, cpu_running_var,
                       momentum, eps, cpu_counts):
        count_sum = torch.sum(cpu_counts)
        cpu_sum = torch.sum(cpu_sum, dim=[0])
        cpu_square_sum = torch.sum(cpu_square_sum, dim=[0])
        cpu_mean = cpu_sum / count_sum
        cpu_square_mean = cpu_square_sum / count_sum
        cpu_var = cpu_square_mean - cpu_mean * cpu_mean

        if cpu_var.dtype != torch.float32:
            cpu_invstd = 1 / torch.sqrt(cpu_var.float() + eps)
            cpu_invstd = cpu_invstd.half()
        else:
            cpu_invstd = 1 / torch.sqrt(cpu_var + eps)

        cpu_running_mean = momentum * cpu_mean + (1 - momentum) * cpu_running_mean
        cpu_running_var = momentum * cpu_var + (1 - momentum) * cpu_running_var
        cpu_mean = cpu_mean.cpu().numpy()
        cpu_invstd = cpu_invstd.cpu().numpy()
        cpu_running_mean = cpu_running_mean.cpu().numpy()
        cpu_running_var = cpu_running_var.cpu().numpy()
        return cpu_mean, cpu_invstd, cpu_running_mean, cpu_running_var

    def npu_op_exec(self, *args):
        npu_mean, npu_invstd = torch_npu.batch_norm_gather_stats_update(*args)
        out_mean = npu_mean.cpu().numpy()
        out_invstd = npu_invstd.cpu().numpy()
        return out_mean, out_invstd

    def create_counts_tensor(self, item):
        dtype = item[0]
        npu_format = item[1]
        if dtype != torch.float32:
            data = [4, 5, 3, 2]
        else:
            data = [4, 5, 6, 4]
        input1 = np.array(data).astype(dtype)
        cpu_counts = torch.from_numpy(input1)
        npu_counts = cpu_counts.to("npu:0")
        if npu_format != -1:
            npu_counts = torch_npu.npu_format_cast(npu_counts, npu_format)
        return cpu_counts, npu_counts

    def test_batch_norm_gather_stats_update(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float32, -1, [2, 3, 12, 12]], [np.float32, -1, [4, 3]], [np.float32, -1, [4, 3]],
             [np.float32, -1, [3]], [np.float32, -1, [3]], 1e-3, 1e-5, [np.float32, -1, [4]]],
            [[np.float16, -1, [16, 3, 12, 12]], [np.float16, -1, [4, 3]], [np.float16, -1, [4, 3]],
             [np.float16, -1, [3]], [np.float16, -1, [3]], 1e-2, 1e-4, [np.float16, -1, [4]]],
        ]

        for item in shape_format:
            cpu_counts, npu_counts = self.create_counts_tensor(item[7])
            cpu_input1, npu_input1fp16 = create_common_tensor(item[0], 1, 10)
            if item[1][0] == np.float32:
                npu_input1fp32 = npu_input1fp16.float()
            cpu_sum, npu_sum = create_common_tensor(item[1], 0, 1)
            cpu_square_sum, npu_square_sum = create_common_tensor(item[2], 0, 1)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 0, 1)
            cpu_running_var, npu_running_var = create_common_tensor(item[4], 0, 1)

            cpu_output = self.expect_cpu_out(cpu_sum, cpu_square_sum,
                                             cpu_running_mean, cpu_running_var,
                                             item[-3], item[-2], cpu_counts)

            npu_outputfp16 = self.npu_op_exec(npu_input1fp16, npu_sum, npu_square_sum,
                                              npu_running_mean, npu_running_var,
                                              item[-3], item[-2], npu_counts)

            self.assertRtolEqual(npu_outputfp16[0], cpu_output[0])
            self.assertRtolEqual(npu_outputfp16[1], cpu_output[1])
            self.assertRtolEqual(npu_running_mean.cpu().numpy(), cpu_output[2])
            self.assertRtolEqual(npu_running_var.cpu().numpy(), cpu_output[3])

            if item[1][0] == np.float32:
                npu_outputfp32 = self.npu_op_exec(npu_input1fp32, npu_sum, npu_square_sum,
                                                  npu_running_mean, npu_running_var,
                                                  item[-3], item[-2], npu_counts)
                self.assertRtolEqual(npu_outputfp32[0], cpu_output[0])
                self.assertRtolEqual(npu_outputfp32[1], cpu_output[1])


if __name__ == "__main__":
    run_tests()
