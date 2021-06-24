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

class TestMean(TestCase):
    def cpu_op_exec(self, input1, dim, keepdim):
        output = torch.mean(input1, dim, keepdim=keepdim)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dim, keepdim):
        output = torch.mean(input1, dim, keepdim=keepdim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def npu_op_exec_out(self, input1, dim, keepdim, input2):
        torch.mean(input1, dim, keepdim=keepdim, out=input2)
        output = input2.to("cpu")
        output = output.numpy()
        return output

    def mean_result_dim_out(self, shape_format):
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_input2, npu_input2 = create_common_tensor(item[0], -100, 100)
            cpu_input3, npu_input3 = create_common_tensor(item[2], -100, 100)
            if cpu_input1.dtype == torch.float16:
                cpu_input1 = cpu_input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[3])
            npu_output_out1 = self.npu_op_exec_out(npu_input1, item[1], item[3], npu_input2)
            npu_output_out2 = self.npu_op_exec_out(npu_input1, item[1], item[3], npu_input3)
            cpu_output1 = cpu_output.astype(npu_output_out1.dtype)
            cpu_output2 = cpu_output.astype(npu_output_out2.dtype)
            self.assertRtolEqual(cpu_output1, npu_output_out1)
            self.assertRtolEqual(cpu_output2, npu_output_out2)

    def test_mean_dim_shape_format_fp16_3d(self, device):
        format_list = [0]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64]], [0, 1], [np.float32, i, [18, 256, 64]], j] for i in format_list for j in
                        keepdim_list
                        ]
        self.mean_result_dim_out(shape_format)

    def test_mean_dim_shape_format_fp32_3d(self, device):
        format_list = [0]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64]], [0, 1], [np.float32, i, [9, 256, 128]], j] for i in format_list for j in
                        keepdim_list
                        ]
        self.mean_result_dim_out(shape_format)

    def test_mean_dim_shape_format_fp16_4d(self, device):
        format_list = [0]
        keepdim_list = [True, False]
        shape_format = [[[np.float16, i, [18, 256, 64, 34]], [0, 1], [np.float32, i, [18, 256, 32]], j] for i in format_list for j
                        in keepdim_list
                        ]
        self.mean_result_dim_out(shape_format)

    def test_mean_dim_shape_format_fp32_4d(self, device):
        format_list = [0]
        keepdim_list = [True, False]
        shape_format = [[[np.float32, i, [18, 256, 64, 34]], [0, 1], [np.float32, i, [18, 256, 128]], j] for i in format_list for j
                        in keepdim_list
                        ]
        self.mean_result_dim_out(shape_format)
    
    def test_mean_null_input(self, device):
        format_list = [0, ]
        shape_list = [(0, 2), (2, 0, 5)]
        dtype_list = [np.float32, ]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, 0, 100)
            cpu_output = cpu_input1.mean()
            npu_output = npu_input1.mean()
            assert torch.isnan(cpu_output)
            assert torch.isnan(npu_output)

instantiate_device_type_tests(TestMean, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
