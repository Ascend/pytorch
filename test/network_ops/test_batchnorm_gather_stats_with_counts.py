import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNormGatherStatsWithCounts(TestCase):
    
    def cpu_op_exec(self, inputs):
        mean, invstd, running_mean, running_var, momentum, eps, counts = inputs
        mean = mean.float()
        invstd = invstd.float()
        running_mean = running_mean.float().unsqueeze(0)
        running_var = running_var.float().unsqueeze(0)
        counts = counts.float().unsqueeze(-1)
        count_broadcast = torch.broadcast_to(counts, invstd.shape)
        counts_all_sum = torch.sum(count_broadcast, [0, ], keepdim=True)

        sum_out_broadcast = counts_all_sum.expand(count_broadcast.shape)
        x_count = mean.mul(count_broadcast)
        x_mean = x_count.div(sum_out_broadcast)
        mean_all = torch.sum(x_mean, [0, ], keepdim=True)

        mean_broadcast = mean_all.expand(mean.shape)
        var_all = torch.reciprocal(invstd)
        var_all_square = torch.mul(var_all, var_all)
        var_all_square_epsilon = torch.add(var_all_square, -eps)
        mean_sub = torch.sub(mean, mean_broadcast)
        mean_var = torch.mul(mean_sub, mean_sub)
        mean_var_sum = torch.add(var_all_square_epsilon, mean_var)
        mean_var_count = torch.mul(mean_var_sum, count_broadcast)
        var_sum = torch.sum(mean_var_count, [0, ], keepdim=True)
        var_sum_count = torch.div(var_sum, counts_all_sum)
        var_sum_count_epsilon = torch.add(var_sum_count, eps)
        var_sqrt = var_sum_count_epsilon.sqrt()
        invert_std = torch.reciprocal(var_sqrt)
        count_sum_one = torch.add(counts_all_sum, torch.tensor([-1], dtype=torch.float32))
        unbiased_var = torch.div(var_sum, count_sum_one)
        running_var_tmp = torch.mul(running_var, 1 - momentum)
        running_var_update_tmp = torch.mul(unbiased_var, momentum)
        running_var_update = torch.add(running_var_tmp, running_var_update_tmp)

        mean_temp = torch.mul(mean_all, momentum)
        running_mean_mul = torch.mul(running_mean, momentum)
        running_mean_temp = torch.sub(running_mean, running_mean_mul)
        running_mean_update = torch.add(mean_temp, running_mean_temp)

        mean_all.squeeze_(0)
        invert_std.squeeze_(0)
        running_mean_update.squeeze_(0)
        running_var_update.squeeze_(0)
        return [mean_all.numpy(), invert_std.numpy(), running_mean_update.numpy(), running_var_update.numpy()]

    def npu_op_exec(self, *args):
        npu_mean, npu_invstd = torch.batch_norm_gather_stats_with_counts(*args)
        out_mean = npu_mean.cpu().numpy()
        out_invstd = npu_invstd.cpu().numpy()
        return out_mean, out_invstd

    def create_counts_tensor(self, item, index):
        dtype = item[0]
        data = [[4, 5, 6, 4], [4, 5, 3, 2]]
        input1 = np.array(data[index]).astype(dtype)
        cpu_counts = torch.from_numpy(input1)
        return cpu_counts

    def test_batch_norm_gather_stats_with_counts(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float16, -1, [2, 3, 12, 12]], [np.float32, -1, [4, 3]], [np.float32, -1, [4, 3]],
             [np.float32, -1, [3]], [np.float32, -1, [3]], 1e-3, 1e-5, [np.float32, -1, [4]], 0],
            [[np.float16, -1, [16, 3, 12, 12]], [np.float16, -1, [4, 3]], [np.float16, -1, [4, 3]],
             [np.float16, -1, [3]], [np.float16, -1, [3]], 1e-2, 1e-4, [np.float16, -1, [4]], 1],
        ]

        for item in shape_format:
            cpu_counts = self.create_counts_tensor(item[7], item[8])
            assert item[-2][2][0] == item[1][-1][0]
            _, npu_input1fp16 = create_common_tensor(item[0], 1, 10)
            if item[1][0] == np.float32:
                npu_input1fp32 = npu_input1fp16.float()
            cpu_mean, npu_mean = create_common_tensor(item[1], 0, 1)
            cpu_invstd, npu_invstd = create_common_tensor(item[2], 0, 1)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 0, 1)
            cpu_running_invstd, npu_running_invstd = create_common_tensor(item[4], 0, 1)
            cpu_output = self.cpu_op_exec([cpu_mean, cpu_invstd,
                                          cpu_running_mean, cpu_running_invstd,
                                          item[-4], item[-3], cpu_counts])
            npu_outputfp16 = self.npu_op_exec(npu_input1fp16, npu_mean, npu_invstd,
                                              npu_running_mean, npu_running_invstd,
                                              item[-4], item[-3], cpu_counts.npu())

            self.assertRtolEqual(npu_outputfp16[0], cpu_output[0].astype(item[1][0]))
            self.assertRtolEqual(npu_outputfp16[1], cpu_output[1].astype(item[1][0]))
            self.assertRtolEqual(npu_running_mean.cpu().numpy(), cpu_output[2].astype(item[1][0]))
            self.assertRtolEqual(npu_running_invstd.cpu().numpy(), cpu_output[3].astype(item[1][0]))

            if item[1][0] == np.float32:
                npu_outputfp32 = self.npu_op_exec(npu_input1fp32, npu_mean, npu_invstd,
                                                  npu_running_mean, npu_running_invstd,
                                                  item[-4], item[-3], cpu_counts.npu())
                self.assertRtolEqual(npu_outputfp32[0], cpu_output[0])
                self.assertRtolEqual(npu_outputfp32[1], cpu_output[1])


if __name__ == "__main__":
    run_tests()
