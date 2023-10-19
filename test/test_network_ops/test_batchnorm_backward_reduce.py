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


class TestBatchNormBackwardReduce(TestCase):
    def expect_result(self):
        cpu_output0 = np.array([449.18185, 464.78906, 471.87485], dtype=np.float32)
        cpu_output1 = np.array([831.08484, 2112.0908, 259.91568], dtype=np.float32)
        cpu_output2 = np.array([6091.88, 3367.45, 1824.8948], dtype=np.float32)
        cpu_output3 = np.array([449.18185, 464.78906, 471.87485], dtype=np.float32)
        list1 = [cpu_output0, cpu_output1, cpu_output2, cpu_output3]
        return list1

    def npu_op_exec(self, *args):
        npu_sum_dy, npu_sum_dy_xmu, npu_grad_weight, npu_grad_bias = \
            torch.batch_norm_backward_reduce(*args)
        list2 = [npu_sum_dy.cpu().numpy(), npu_sum_dy_xmu.cpu().numpy(),
                 npu_grad_weight.cpu().numpy(), npu_grad_bias.cpu().numpy()]
        return list2

    def test_batch_norm_backward_reduce_mix_precision(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float16, -1, [1, 3, 9, 9]], [np.float32, -1, [3]],
             True, True, True],
        ]
        for item in shape_format:
            _, npu_grad_output_fp16 = create_common_tensor(item[0], 1, 10)
            _, npu_input1_fp16 = create_common_tensor(item[0], 1, 10)
            _, npu_mean = create_common_tensor(item[1], 1, 10)
            _, npu_invstd = create_common_tensor(item[1], 1, 10)
            _, npu_weight = create_common_tensor(item[1], 1, 10)
            npu_grad_output_fp32 = npu_grad_output_fp16.float()
            npu_input1_fp32 = npu_input1_fp16.float()

            npu_output_fp16 = self.npu_op_exec(npu_grad_output_fp16,
                                               npu_input1_fp16, npu_mean,
                                               npu_invstd, npu_weight,
                                               *item[-3:])
            npu_output_fp32 = self.npu_op_exec(npu_grad_output_fp32,
                                               npu_input1_fp32, npu_mean,
                                               npu_invstd, npu_weight,
                                               *item[-3:])
            for out16, out32 in zip(npu_output_fp16, npu_output_fp32):
                self.assertRtolEqual(out16, out32)

    def test_batch_norm_backward_reduce(self):
        np.random.seed(1234)
        shape_format = [
            [[np.float32, -1, [1, 3, 9, 9]], [np.float32, -1, [3]],
             True, True, True],
        ]
        for item in shape_format:
            _, npu_grad_output = create_common_tensor(item[0], 1, 10)
            _, npu_input1 = create_common_tensor(item[0], 1, 10)
            _, npu_mean = create_common_tensor(item[1], 1, 10)
            _, npu_invstd = create_common_tensor(item[1], 1, 10)
            _, npu_weight = create_common_tensor(item[1], 1, 10)

            list1 = self.expect_result()
            list2 = self.npu_op_exec(npu_grad_output,
                                     npu_input1, npu_mean,
                                     npu_invstd, npu_weight,
                                     *item[-3:])

            self.assertRtolEqual(list1[0], list2[0])
            self.assertRtolEqual(list1[1], list2[1])
            self.assertRtolEqual(list1[2], list2[2])
            self.assertRtolEqual(list1[3], list2[3])


if __name__ == "__main__":
    run_tests()
