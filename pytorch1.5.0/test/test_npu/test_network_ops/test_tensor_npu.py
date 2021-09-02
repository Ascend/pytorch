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
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor

class TestTensorNpu(TestCase):
    
    def cpu_op_exec(self, input):
        output = input.to("cpu")
        return output

    def npu_op_exec(self, input):
        output = input.npu()
        output = output.to("cpu")
        return output

    def cpu_type_exec(self, input):
        output = input.to("cpu")
        output = output.is_npu
        return output

    def npu_type_exec(self, input):
        output = input.npu()
        output = output.is_npu
        return output

    def test_tensor_npu_shape_format(self, device):
        shape_format = [
                [np.float32, 0, 1],
                [np.float32, 0, (64, 10)],
                [np.float32, 3, (256, 2048, 7, 7)],
                [np.float32, 4, (32, 1, 3, 3)],
                [np.float32, 29, (10, 128)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_op_exec(cpu_input)
            npu_output = self.npu_op_exec(npu_input)
            self.assertRtolEqual(cpu_output, npu_output.cpu())

    def test_is_npu_shape_format(self, device):
        shape_format = [
                [np.float32, 0, 1],
                [np.float32, 0, (64, 10)],
                [np.float32, 3, (256, 2048, 7, 7)],
                [np.float32, 4, (32, 1, 3, 3)],
                [np.float32, 29, (10, 128)]
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 1, 100)
            cpu_output = self.cpu_type_exec(cpu_input)
            npu_output = self.npu_type_exec(npu_input)
            self.assertEqual(cpu_output, False)
            self.assertEqual(npu_output, True)

instantiate_device_type_tests(TestTensorNpu, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
