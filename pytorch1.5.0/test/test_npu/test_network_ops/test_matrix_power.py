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
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMatrixPower(TestCase):
    def cpu_op_exec(self, input1, n):
        input1 = input1.float()
        output = torch.matrix_power(input1, n)
        output = output.half()
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, n):
        output = torch.matrix_power(input1, n)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_matrix_power_shape_format(self, device):
        shape_format = [
            [[np.float16, -1, (3, 3)], 0],
            [[np.float16, -1, (3, 3)], 1],
            [[np.float16, -1, (3, 3)], 5],
            [[np.float16, -1, (7, 3, 3)], 1],
            [[np.float16, -1, (2, 5, 5)], 2],
        ]

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], -2, 2)
            cpu_output = self.cpu_op_exec(cpu_input, item[1])
            npu_output = self.npu_op_exec(npu_input, item[1])
            self.assertRtolEqual(cpu_output, npu_output, prec16=0.05)


instantiate_device_type_tests(TestMatrixPower, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
