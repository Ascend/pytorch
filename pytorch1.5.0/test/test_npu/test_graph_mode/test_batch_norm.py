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
from graph_utils import RunFuncInGraphMode
#affine = False,目前测试报错。所以本UT未做affine=False测试
class TestBatchNorm(TestCase):
    def cpu_op_exec(self, input1, num_features, affine):
        flag = False
        if input1.dtype == torch.float16:
            input1 = input1.to(torch.float32)
            flag = True
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        output = m(input1)
        if flag:
            output = output.to(torch.float16)
        output_cpu = output.detach().numpy()
        return output_cpu

    def npu_op_exec_new(self, input1, num_features,affine):
        m = torch.nn.BatchNorm2d(num_features, affine=affine)
        m = m.to("npu")
        output = m(input1)
        output = output.to("cpu").detach().numpy()
        return output
    
    @RunFuncInGraphMode
    def test_batchnorm_shape_format(self, device):
        shape_format = [
                [[np.float32, -1, (10, 32, 35, 45)], True],
                [[np.float32, -1, (256, 100, 7, 7)], True],
                [[np.float32, -1, (256, 100, 14, 14)], True],
                [[np.float32, -1, (10, 56, 28, 28)], True],
                [[np.float32,  0, (10, 50, 14, 14)], True],
                [[np.float32,  3, (10, 24, 50, 50)], True],
                [[np.float32,  3, (10, 56, 56, 56)], True],
                [[np.float32,  3, (10, 100, 7, 7)], True],
                [[np.float32, -1, (10, 10, 28, 28)], True],
                [[np.float32, -1, (10, 150, 28, 28)], True],
                [[np.float32, -1, (10, 200, 7, 7)], True],
                [[np.float32, -1, (10, 100, 14, 14)], True],
                [[np.float16, -1, (256, 100, 7, 7)], True],
                [[np.float16, -1, (256, 100, 14, 14)], True],
                [[np.float16, -1, (10, 56, 28, 28)], True],
                [[np.float16,  0, (10, 50, 14, 14)], True],
                [[np.float16,  3, (10, 24, 50, 50)], True],
                [[np.float16,  3, (10, 56, 56, 56)], True],
                [[np.float16,  3, (10, 100, 7, 7)], True],
                [[np.float16, -1, (10, 10, 28, 28)], True],
                [[np.float16, -1, (10, 150, 28, 28)], True],
                [[np.float16, -1, (10, 200, 7, 7)], True],
                [[np.float16, -1, (10, 100, 14, 14)], True]
                ]
        
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 10)
            cpu_output = self.cpu_op_exec(cpu_input1, item[0][2][1], item[1])
            npu_output = self.npu_op_exec_new(npu_input1, item[0][2][1], item[1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestBatchNorm, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()