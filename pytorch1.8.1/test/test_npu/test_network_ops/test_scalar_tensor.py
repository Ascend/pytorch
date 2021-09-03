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
import torch.nn as nn
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestScalarTensor(TestCase):
    def cpu_op_exec(self, scalar, dtype):
        output = torch.scalar_tensor(scalar, dtype=dtype, device="cpu")
        output = output.numpy()
        return output

    def npu_op_exec(self, scalar, dtype):
        output = torch.scalar_tensor(scalar, dtype=dtype, device="npu")
        output = output.cpu()
        output = output.numpy()
        return output

    def test_scalar_tensor_shape_format(self, device):
        scalars = [-50, 0, 50]
        dtypes = [torch.float16, torch.float32, torch.int32]
        
        shape_format = [
                [i, j] for i in scalars for j in dtypes 
        ]

        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0], item[1])
            npu_output = self.npu_op_exec(item[0], item[1])

            self.assertEqual(cpu_output, npu_output)


instantiate_device_type_tests(TestScalarTensor, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
