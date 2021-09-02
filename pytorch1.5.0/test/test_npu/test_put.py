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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestPut(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1


    def cpu_op_exec(self, input_x, index, source, accumulate):
        input_x = copy.deepcopy(input_x)
        index = index.to("cpu")
        source = source.to("cpu")
        output = torch.Tensor.put_(input_x, index, source, accumulate)
        output = output.numpy()
        return output


    def npu_op_exec(self, input_x, index, source, accumulate):
        input_x = input_x.to("npu")
        index = index.to("npu")
        source = source.to("npu")
        output = torch.Tensor.put_(input_x, index, source, accumulate)
        output = output.to("cpu")
        output = output.numpy()
        return output


    def test_put_common_shape_format(self, device):
        #pylint:disable=unused-argument
        shape_format = [
                [[np.float32, -1, (4, 3)], [np.int64, -1, (4, 1)], [np.float32, -1, (4)]],
                [[np.float32, -1, (4, 3, 5)], [np.int64, -1, (4, 2)], [np.float32, -1, (4, 2)]],
                [[np.float32, -1, (5, 6, 4, 3)], [np.int64, -1, (8, 2)], [np.float32, -1, (8, 2)]],
                [[np.int32, -1, (2, 4, 3, 8)], [np.int64, -1, (10)], [np.int32, -1, (5, 2)]],
                [[np.int32, -1, (9, 3, 4, 3, 9)], [np.int64, -1, (10, 1)], [np.int32, -1, (10)]],
                [[np.float32, -1, (8, 9, 5, 4, 3, 10)], [np.int64, -1, (9, 1)], [np.float32, -1, (9)]],
                [[np.float32, -1, (5, 5, 5, 6, 4, 3)], [np.int64, -1, (2, 4, 5)], [np.float32, -1, (2, 4, 5)]],
                [[np.float32, -1, (6, 9, 10, 2, 5, 7, 3)], [np.int64, -1, (3, 4, 5)], [np.float32, -1, (3, 4, 5)]]
        ]
        for item in shape_format:
            maxVal = 1
            for i in item[0][2]:
                maxVal = maxVal * i
            accumulate = True
            if np.random.randint(0,999) % 2 == 1:
                accumulate = False

            input_x = self.generate_single_data(1, 100, item[0][2], item[0][0])
            index = self.generate_single_data(0, maxVal, item[1][2], item[1][0])
            source = self.generate_single_data(1, 100, item[2][2], item[2][0])
            cpu_output = self.cpu_op_exec(input_x, index, source, accumulate)
            npu_output = self.npu_op_exec(input_x, index, source, accumulate)
            
            self.assertRtolEqual(cpu_output, npu_output)  


instantiate_device_type_tests(TestPut, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
