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

class TestIm2colBackward(TestCase):
    def cpu_op_exec(self,input1, ksizes, strides, dilates, padding):
        input1.requires_grad = True
        output = torch._C._nn.im2col(input1, ksizes, dilates, padding, strides)
        d = output.sum()
        d.backward(retain_graph=True)
        output1 = d.detach().numpy()
        return output1
    def npu_op_exec(self, input1, ksizes, strides, dilates,padding):
        input1.requires_grad = True
        input1 = input1.to("npu")
        output = torch._C._nn.im2col(input1, ksizes, dilates, padding, strides)
        d = output.sum()
        d.backward(retain_graph=True)
        output1 = d.cpu().detach().numpy()
        return output1

    def test_im2col_backward_shape_format(self, device):
        shape_format = [
                [[np.float32, 0, (1, 2, 4, 12)], (2,2), (1,1), (1,1), (0,0)],
                [[np.float32, 3, (1, 2, 8, 30)], (2,2), (1,1), (1,1), (1,1)],
                [[np.float32, 4, (1, 256, 12, 5)], (2,2), (1,1), (1,1), (0,0)],
                [[np.float32, 29, (1, 2048, 12, 12)],(2,2), (1,1), (1,1), (0,0)]
                ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 1, 20)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(cpu_input, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestIm2colBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
