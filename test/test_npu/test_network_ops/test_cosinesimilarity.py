# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestCosinesimilarity(TestCase):

    def generate_data(self, min_num, max_num, shape, dtype):
        input1 = np.random.uniform(min_num, max_num, shape).astype(dtype)
        input2 = np.random.uniform(min_num, max_num, shape).astype(dtype)

        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def cpu_op_exec(self, input_x1, input_x2, dim=1, eps=1e-8):
        cos = torch.nn.CosineSimilarity(dim, eps)
        res = cos(input_x1, input_x2)
        res=res.numpy()
        return res

    def npu_op_exec(self, input1, input2, dim=1, eps=1e-8):
        input1 = input1.npu()
        input2 = input2.npu()
        cos = torch.nn.CosineSimilarity(dim, eps)
        output = cos(input1, input2)
        output = output.cpu()
        output = output.numpy()
        return output

    def test_cosine_similarity(self, device):
        shape_format = [
            [-100, 100, (16, 32), np.float32],
            [-100, 100, (2, 4, 8), np.float32],
            [-100, 100, (2, 4, 6, 8), np.float32],
            [-100, 100, (2, 4, 6, 8, 10), np.float32],
            [-100, 100, (2, 4, 6, 8, 10, 12), np.float32],
            [-0.000030517578125, 0.000030517578125, (2, 32, 149, 31), np.float32],
            [-9.313225746154785e-10, 9.313225746154785e-10, (184965, 1), np.float32],
            [-2, 2, (65535, 1, 1, 1), np.float32],
            [-2, 2, (1, 1, 1, 8192), np.float32],
            [-2, 2, (1, 1, 1, 16384), np.float32],
            [-2, 2, (1, 1, 1, 32768), np.float32],
            [-2, 2, (1, 1, 1, 65535), np.float32],
            [-2, 2, (1, 1, 1, 131072), np.float32],
            [-2, 2, (1, 1, 1, 196608), np.float32],
            [-2, 2, (1, 1, 1, 262144), np.float32],
            [-2, 2, (1, 1, 1, 393216), np.float32],
            [-2, 2, (1, 1, 1, 524288), np.float32],
            [-2, 2, (1, 1, 1, 655360), np.float32],
            [-2, 2, (1, 1, 1, 786432), np.float32],
            [0, 0, (2, 4, 16), np.float32],
            ]
            
        def test_cosinesimilarity_float32(self, min, max, shape, dtype, dim=1, eps=1e-8):
            cpu_input1, cpu_input2 = self.generate_data(min, max, shape, dtype)
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_input2, dim=dim, eps=eps)
            npu_output = self.npu_op_exec(cpu_input1, cpu_input2, dim=dim, eps=eps)
            self.assertRtolEqual(cpu_output, npu_output)
        for item in shape_format:
            test_cosinesimilarity_float32(self, item[0], item[1], item[2], item[3])
    
instantiate_device_type_tests(TestCosinesimilarity, globals(), except_for='cpu')
if __name__ == '__main__':
    run_tests()
