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
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestCastLong(TestCase):

    def generate_data(self, minValue, maxValue, shape, dtype):
        input1 = np.random.uniform(minValue, maxValue, shape).astype(dtype)
        # modify from numpy.ndarray to torch.tensor
        npu_input1 = torch.from_numpy(input1)
        return npu_input1


    def cpu_op_exec(self, input1):
        output = torch._cast_Long(input1)
        output = output.numpy()
        return output


    def npu_op_exec(self, input1):
        input1 = input1.to("npu")
        output = torch._cast_Long(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output


    def test_cast_Long_common_shape_format(self, device):
        shape_format = [
            [[np.bool, -1, (4, 3, 1)]],
            [[np.int32, -1, (4, 3, 1)]],
            [[np.int8, -1, (2, 3)]],
            [[np.float32, -1, (4, 3, 1)]],
            [[np.float16, -1, (4, 3, 1)]],
            [[np.uint8, -1, (4, 3, 1)]]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            cpu_output = self.cpu_op_exec(cpu_input1)
            npu_output = self.npu_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestCastLong, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()