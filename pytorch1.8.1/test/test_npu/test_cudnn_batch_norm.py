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
#from torch.testing._internal.common_utils import TestCase, run_tests
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCudnnBatchNorm(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        
        return npu_input1, npu_input2
        

    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        
        return npu_input1

    def generate_three_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input2 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input3 = np.random.uniform(min_d, max_d, shape).astype(dtype)

        #modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)
        npu_input3 = torch.from_numpy(input3)
        
        return npu_input1, npu_input2, npu_input3

    def generate_scalar(self, min_d, max_d):
        scalar = np.random.uniform(min_d, max_d)
        return scalar

    def generate_int_scalar(self, min_d, max_d):
        scalar = np.random.randint( min_d, max_d)
        return scalar


    def cpu_op_exec(self, input1, weight, cpu_bias,cpu_running_mean, cpu_running_var, traning, exponential_average_factor, epsilon):
        output = torch.batch_norm(input1, weight, cpu_bias, cpu_running_mean, cpu_running_var, traning, exponential_average_factor, epsilon, cudnn_enabled = False)
        return output 

    def npu_op_exec(self, input1, weight, npu_bias,npu_running_mean, npu_running_var, traning, exponential_average_factor, epsilon):
        output,_,_,_ = torch.cudnn_batch_norm(input1, weight, npu_bias,npu_running_mean, npu_running_var, traning, exponential_average_factor, epsilon)
        output = output.to("cpu")
        return output
    
    def test_cudnn_batch_norm_shape_format(self, device):
        shape_format = [
               # [[np.int32, 0, (2,  2, 4, 1)], [np.int32, 0, (2)], [np.int32, 0, (2)], [np.int32, 0, (2)], [np.int32, 0, (2)]],
                [[np.float32, 0,  (20,  30,  40, 20)], [np.float32, 0,  (30)], [np.float32, 0,  (30)], [np.float32, 0,  (30)], [np.float32,  0, (30)]],
                # [[np.float16, 0,  (2,  2,  3, 2)], [np.float16, 0,  (2)], [np.float16, 0,  (2)], [np.float16, 0,  (2)], [np.float16,  0, (2)]],
                #[[np.float16, 3, (2,  20, 2, 3)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)], [np.float16, 3, (20)]],
                [[np.float32, 3, (2,  20, 2, 3)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)], [np.float32, 3, (20)]]
                ]
        
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 100)
            cpu_input_weight, npu_input_weight = create_common_tensor(item[1],  1, 10)
            cpu_bias, npu_bias = create_common_tensor(item[2]  , 1, 10)
            cpu_running_mean, npu_running_mean = create_common_tensor(item[3], 1, 10)
            cpu_running_var, npu_running_var = create_common_tensor(item[4]  , 1, 10)
            cpu_result = self.cpu_op_exec(cpu_input, cpu_input_weight, cpu_bias,cpu_running_mean, cpu_running_var, False, 0.1, 0)
            npu_result = self.npu_op_exec(npu_input, npu_input_weight, npu_bias, npu_running_mean, npu_running_var, False, 0.1, 0)
            self.assertRtolEqual(cpu_result, npu_result);
instantiate_device_type_tests(TestCudnnBatchNorm, globals(), except_for="cpu")
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
