# Copyright (c) 2020 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION. 
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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.decorator import graph_mode
from torch_npu.testing.common_utils import create_common_tensor


class TestMean(TestCase):
    
    def cpu_op_exec(self,input1, dtype):
        output = torch.mean(input1, [2, 3], keepdim=True, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self,input1, dtype):
        input1 = input1.to("npu")
        output = torch.mean(input1, [2, 3], keepdim=True, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output
    
    @graph_mode
    def test_mean_shape_format(self):
        shape_format = [
                 [[np.float32, 3, (256, 1280, 7, 7)],  torch.float32],
                 [[np.float16, 3, (1024, 1024, 7, 7)], torch.float32],

        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, dtype=item[-1])
            npu_output = self.npu_op_exec(npu_input, dtype=item[-1])
            self.assertRtolEqual(cpu_output, npu_output)
    
if __name__ == "__main__":
    run_tests()