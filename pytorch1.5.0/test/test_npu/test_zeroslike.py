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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestZerosLike(TestCase):
    def cpu_op_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, dtype):
        output = torch.zeros_like(input1, dtype=dtype)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_zeroslike_shape_format(self, device):
        shape_format = [
            [[np.float32, 4, [32, 144, 1, 1]], torch.float32],
            [[np.float32, 3, [16]], torch.float32],
            [[np.float32, 0, [960, 1, 3, 3]], torch.float32],
            [[np.float32, 29, [1000, 1280]], torch.float32],
            [[np.float16, 4, [32, 128, 3, 3]], torch.float16],
            [[np.float16, 29, [1000, 1024]], torch.float16],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestZerosLike, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
