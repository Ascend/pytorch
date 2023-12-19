# Copyright (c) 2023, Huawei Technologies.All rights reserved.
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
from torch_npu.testing.common_utils import create_common_tensor


class TestAddcMul(TestCase):

    def test_add_scalar_shape_format_fp32(self):
        shape_format = [
            [np.float32, 2, [50]],
            [np.float32, 2, [50, 25]],
            [np.float32, 2, [50, 25, 7]],
            [np.float32, 2, [50, 25, 7, 100]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item, -10, 10)
            cpu_input3, npu_input3 = create_common_tensor(item, -10, 10)
            cpu_output1 = torch._foreach_addcmul([cpu_input1], [cpu_input2], [cpu_input3], 1.0)
            npu_output1 = torch._foreach_addcmul([npu_input1], [npu_input2], [npu_input3], 1.0)
            cpu_output2 = torch._foreach_addcmul([cpu_input1], [cpu_input2], [cpu_input3], [1.0])
            npu_output2 = torch._foreach_addcmul([npu_input1], [npu_input2], [npu_input3], [1.0])
            for (cpu_tmp1, npu_tmp1) in zip(cpu_output1, npu_output1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            for (cpu_tmp2, npu_tmp2) in zip(cpu_output2, npu_output2):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())

    def test_add_scalar_shape_format_fp32_(self):
        shape_format = [
            [np.float32, 2, [50]],
            [np.float32, 2, [50, 25]],
            [np.float32, 2, [50, 25, 7]],
            [np.float32, 2, [50, 25, 7, 100]],
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item, -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item, -10, 10)
            cpu_input3, npu_input3 = create_common_tensor(item, -10, 10)
            torch._foreach_addcmul_([cpu_input1], [cpu_input2], [cpu_input3], 1.0)
            torch._foreach_addcmul_([npu_input1], [npu_input2], [npu_input3], 1.0)

            for (cpu_tmp1, npu_tmp1) in zip(cpu_input1, npu_input1):
                self.assertRtolEqual(cpu_tmp1.numpy(), npu_tmp1.to("cpu").numpy())

            cpu_input1, npu_input1 = create_common_tensor(item, -10, 10)
            cpu_input2, npu_input2 = create_common_tensor(item, -10, 10)
            cpu_input3, npu_input3 = create_common_tensor(item, -10, 10)
            torch._foreach_addcmul_([cpu_input1], [cpu_input2], [cpu_input3], [1.0])
            torch._foreach_addcmul_([npu_input1], [npu_input2], [npu_input3], [1.0])

            for (cpu_tmp2, npu_tmp2) in zip(cpu_input1, npu_input1):
                self.assertRtolEqual(cpu_tmp2.numpy(), npu_tmp2.to("cpu").numpy())


if __name__ == "__main__":
    run_tests()
