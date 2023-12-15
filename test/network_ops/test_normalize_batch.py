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

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestNormalizeBatch(TestCase):
    def cpu_op_exec(self, input1, seq_len, normalize_type):
        if normalize_type == 0:
            x_mean = torch.zeros((seq_len.shape[0], input1.shape[1]), dtype=input1.dtype,
                                 device=input1.device)
            x_std = torch.zeros((seq_len.shape[0], input1.shape[1]), dtype=input1.dtype,
                                device=input1.device)
            for i in range(input1.shape[0]):
                x_mean[i, :] = input1[i, :, :seq_len[i]].mean(dim=1)
                x_std[i, :] = input1[i, :, :seq_len[i]].std(dim=1)
            x_std += 1e-5
            result = (input1 - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        else:
            x_mean = torch.zeros(seq_len.shape, dtype=input1.dtype,
                                 device=input1.device)
            x_std = torch.zeros(seq_len.shape, dtype=input1.dtype,
                                device=input1.device)
            for i in range(input1.shape[0]):
                x_mean[i] = input1[i, :, :int(seq_len[i])].mean()
                x_std[i] = input1[i, :, :int(seq_len[i])].std()
            x_std += 1e-5
            result = (input1 - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        return result.numpy()

    def npu_op_exec(self, input1, seq_len, normalize_type):
        out = torch_npu.npu_normalize_batch(input1, seq_len, normalize_type)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_normalize_batch(self, device="npu"):
        shape_format = [
            [[np.float32, -1, [32, 3, 6]], [np.int32, -1, [32]], 0],
            [[np.float32, -1, [32, 3, 6]], [np.int32, -1, [32]], 1],
            [[np.float32, -1, [16, 6, 1000]], [np.int32, -1, [16]], 0],
            [[np.float32, -1, [16, 6, 1000]], [np.int32, -1, [16]], 1]
        ]
        for item in shape_format:
            right_range = item[0][-1][-1]
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 10)
            cpu_seqlen, npu_seqlen = create_common_tensor(item[1], 3, right_range)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_seqlen, item[-1])
            npu_output = self.npu_op_exec(npu_input1, npu_seqlen, item[-1])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
