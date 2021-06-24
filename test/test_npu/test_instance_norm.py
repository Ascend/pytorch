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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestIn2d(TestCase):
    def cpu_op_exec(self, input1, weight, cpu_bias, cpu_running_mean, cpu_running_var, use_input_stats, momentum, epsilon):
        output = torch.instance_norm(input1, weight, cpu_bias, cpu_running_mean, cpu_running_var, use_input_stats, momentum, epsilon, cudnn_enabled = False)
        return output.numpy()

    def npu_op_exec(self, input1, weight, npu_bias, npu_running_mean, npu_running_var, use_input_stats, momentum, epsilon):
        output = torch.instance_norm(input1, weight, npu_bias, npu_running_mean, npu_running_var, use_input_stats, momentum, epsilon, cudnn_enabled = False)
        output = output.to("cpu")
        return output.numpy()

    def test_instance_norm_shape_format(self, device):
        shape_format = [
            [[np.float32, 0, (2, 20, 8, 10)], [np.float32, 0,  (20)], [np.float32, 0,  (20)], [np.float32, 0,  (20)], [np.float32,  0, (20)], False, 0.1, 0.0001],
            [[np.float32, 0, (2, 8, 10, 7)], [np.float32, 0,  (8)], [np.float32, 0,  (8)], [np.float32, 0,  (8)], [np.float32,  0, (8)], True, 0.1, 0.0001],
            [[np.float32, 0, (2, 10, 20)], [np.float32, -1, (10,)], [np.float32, -1, (10,)],[np.float32, -1, (10,)], [np.float32, -1, (10,)], True, 0.1, 0.0001],
            [[np.float32, 3, (6,  20, 2, 3)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)], False, 0.1, 0.0001],
            [[np.float32, 3, (6,  20, 2, 3)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)], True, 0.1, 0.0001],
            [[np.float32, 3, (2, 2, 2, 2)], [np.float32, -1, (2,)], [np.float32, -1, (2,)],[np.float32, -1, (2,)], [np.float32, -1, (2,)], True, 0.1, 0.0001]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 20)
            cpu_input_weight, npu_input_weight = create_common_tensor(item[1],  1, 10)
            cpu_bias, npu_bias = create_common_tensor(item[2], 1, 10)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 1, 10)
            cpu_running_var, npu_running_var = create_common_tensor(item[4], 1, 10)
            cpu_result = self.cpu_op_exec(cpu_input, cpu_input_weight, cpu_bias, cpu_running_mean, cpu_running_var, item[5], item[6], item[7])
            npu_result = self.npu_op_exec(npu_input, npu_input_weight, npu_bias, npu_running_mean, npu_running_var, item[5], item[6], item[7])
            self.assertRtolEqual(cpu_result, npu_result)

    def test_instance_norm_fp16_shape_format(self, device):
        shape_format = [
            [[np.float16, 0, (2, 15, 4, 2)], [np.float16, 0,  (15)], [np.float16, 0,  (15)], [np.float16, 0,  (15)], [np.float16,  0, (15)], False, 0.1, 0.0001],
            [[np.float16, 0, (2, 30, 4, 2)], [np.float16, 0,  (30)], [np.float16, 0,  (30)], [np.float16, 0,  (30)], [np.float16,  0, (30)], True, 0.1, 0.0001],
            [[np.float16, 0, (2, 10, 20)], [np.float16, -1, (10,)], [np.float16, -1, (10,)],[np.float16, -1, (10,)], [np.float16, -1, (10,)], True, 0.1, 0.0001],
            [[np.float16, 3, (6,  20, 2, 3)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)], False, 0.1, 0.0001],
            [[np.float16, 3, (6,  20, 2, 3)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)], True, 0.1, 0.0001],
            [[np.float16, 3, (2, 2, 2, 2)], [np.float16, -1, (2,)], [np.float16, -1, (2,)],[np.float16, -1, (2,)], [np.float16, -1, (2,)], True, 0.1, 0.0001]
        ]
        def cpu_op_fp16_exec(input1,
                             weight,
                             cpu_bias,
                             cpu_running_mean,
                             cpu_running_var,
                             use_input_stats,
                             momentum,
                             epsilon):
            input1 = input1.to(torch.float32)
            weight = weight.to(torch.float32)
            cpu_bias = cpu_bias.to(torch.float32)
            cpu_running_mean = cpu_running_mean.to(torch.float32)
            cpu_running_var = cpu_running_var.to(torch.float32)

            output = torch.instance_norm(input1,
                                         weight,
                                         cpu_bias,
                                         cpu_running_mean,
                                         cpu_running_var,
                                         use_input_stats,
                                         momentum,
                                         epsilon,
                                         cudnn_enabled = False)
            output = output.numpy()
            return output.astype(np.float16)

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input_weight, npu_input_weight = create_common_tensor(item[1],  1, 10)
            cpu_bias, npu_bias = create_common_tensor(item[2], 1, 10)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 1, 10)
            cpu_running_var, npu_running_var = create_common_tensor(item[4], 1, 10)
            cpu_result = cpu_op_fp16_exec(cpu_input, cpu_input_weight, cpu_bias, cpu_running_mean, cpu_running_var, item[5], item[6], item[7])
            npu_result = self.npu_op_exec(npu_input, npu_input_weight, npu_bias, npu_running_mean, npu_running_var, item[5], item[6], item[7])
            self.assertRtolEqual(cpu_result, npu_result)

instantiate_device_type_tests(TestIn2d, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()