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

"""
op_models/92_ReduceMeanD.om/ReduceMeanD_tvmbin ReduceMeanD IN[DT_FLOAT NC1HWC0[128, 128, 7, 7, 16]] OUT[DT_FLOAT NC1HWC0[128, 128, 1, 1, 16]]
"""
import sys
import torch
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor
from graph_utils import RunFuncInGraphMode

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

    @RunFuncInGraphMode
    def test_mean_shape_format(self, device):
        shape_format = [
                 [[np.float32,3,[256, 1280, 7, 7]],torch.float32],
                 [[np.float16,3, [1024, 1024, 7, 7]],torch.float32],

        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, dtype=item[-1])
            npu_output = self.npu_op_exec(npu_input, dtype=item[-1])
            self.assertRtolEqual(cpu_output, npu_output)

instantiate_device_type_tests(TestMean, globals(), except_for="cpu")
if __name__ == "__main__":
     run_tests()


