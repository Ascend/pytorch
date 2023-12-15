# Copyright (c) 2020, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestBatchNormGatherStatsWithCounts(TestCase):
    def expect_cuda_out_fp16(self):
        return [np.array([0.5757, 0.4543, 0.3857], dtype=np.float16),
                np.array([0.139, 0.1124, 0.2357], dtype=np.float16),
                np.array([0.0842, 0.9673, 0.75], dtype=np.float16),
                np.array([0.681, 1.668, 1.11], dtype=np.float16)]

    def expect_cuda_out_fp32(self):
        return [np.array([0.46471214, 0.6849079, 0.83278275], dtype=np.float32),
                np.array([0.3682663, 0.46639538, 0.23710594], dtype=np.float32),
                np.array([0.41927528, 0.56878287, 0.04250176], dtype=np.float32),
                np.array([1.0024216, 0.6232378, 0.7974624], dtype=np.float32)]

    def npu_op_exec(self, *args):
        npu_mean, npu_invstd = torch.batch_norm_gather_stats_with_counts(*args)
        out_mean = npu_mean.cpu().numpy()
        out_invstd = npu_invstd.cpu().numpy()
        return out_mean, out_invstd

    def create_counts_tensor32(self, item):
        dtype = item[0]
        npu_format = item[1]
        shape = item[2]
        data = [4, 5, 6, 4]
        input1 = np.array(data).astype(dtype)
        npu_counts = torch.from_numpy(input1).to("npu:0")
        if npu_format != -1:
            npu_counts = torch_npu.npu_format_cast(npu_counts, npu_format)
        return npu_counts

    def create_counts_tensor16(self, item):
        dtype = item[0]
        npu_format = item[1]
        shape = item[2]
        data = [4, 5, 3, 2]
        input1 = np.array(data).astype(dtype)
        npu_counts = torch.from_numpy(input1).to("npu:0")
        if npu_format != -1:
            npu_counts = torch_npu.npu_format_cast(npu_counts, npu_format)
        return npu_counts

    @unittest.skip("skip test_batch_norm_gather_stats_with_counts now")
    def test_batch_norm_gather_stats_with_counts(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float16, -1, [2, 3, 12, 12]], [np.float32, -1, [4, 3]], [np.float32, -1, [4, 3]],
             [np.float32, -1, [3]], [np.float32, -1, [3]], 1e-3, 1e-5, [np.float32, -1, [4]], 0],
            [[np.float16, -1, [16, 3, 12, 12]], [np.float16, -1, [4, 3]], [np.float16, -1, [4, 3]],
             [np.float16, -1, [3]], [np.float16, -1, [3]], 1e-2, 1e-4, [np.float16, -1, [4]], 1],
        ]

        for item in shape_format:
            if item[8] == 0:
                npu_counts = self.create_counts_tensor32(item[7])
            else:
                npu_counts = self.create_counts_tensor16(item[7])
            assert item[-2][2][0] == item[1][-1][0]
            cpu_input1, npu_input1fp16 = create_common_tensor(item[0], 1, 10)
            if item[1][0] == np.float32:
                npu_input1fp32 = npu_input1fp16.float()
            cpu_mean, npu_mean = create_common_tensor(item[1], 0, 1)
            cpu_invstd, npu_invstd = create_common_tensor(item[2], 0, 1)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 0, 1)
            cpu_running_invstd, npu_running_invstd = create_common_tensor(item[4], 0, 1)

            if item[1][0] == np.float16:
                cuda_output = self.expect_cuda_out_fp16()
            else:
                cuda_output = self.expect_cuda_out_fp32()

            npu_outputfp16 = self.npu_op_exec(npu_input1fp16, npu_mean, npu_invstd,
                                              npu_running_mean, npu_running_invstd,
                                              item[-4], item[-3], npu_counts)
            self.assertRtolEqual(npu_outputfp16[0], cuda_output[0])
            self.assertRtolEqual(npu_outputfp16[1], cuda_output[1])
            self.assertRtolEqual(npu_running_mean.cpu().numpy(), cuda_output[2])
            self.assertRtolEqual(npu_running_invstd.cpu().numpy(), cuda_output[3])

            if item[1][0] == np.float32:
                npu_outputfp32 = self.npu_op_exec(npu_input1fp32, npu_mean, npu_invstd,
                                                  npu_running_mean, npu_running_invstd,
                                                  item[-4], item[-3], npu_counts)
                self.assertRtolEqual(npu_outputfp32[0], cuda_output[0])
                self.assertRtolEqual(npu_outputfp32[1], cuda_output[1])


if __name__ == "__main__":
    run_tests()
