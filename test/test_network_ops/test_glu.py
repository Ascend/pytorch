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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestGlu(TestCase):
    def generate_single_data(self, min_d, max_d, shape, dtype):
        input1 = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input_data, dim):
        input_data = input_data.to("cpu")
        flag = False
        if input_data.dtype == torch.float16:
            input_data = input_data.to(torch.float32)
            flag = True
        output = torch.nn.functional.glu(input_data, dim)

        if flag:
            output = output.to(torch.float16)
        output = output.numpy()
        return output

    def npu_op_exec(self, input_data, dim):
        input_data = input_data.to("npu")
        output = torch.nn.functional.glu(input_data, dim)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_put_common_shape_format(self, device="npu"):
        shape_format = [
            [np.float32, (4, 8), -1, 100, 200],
            [np.float32, (4, 6, 8), -2, 100, 200],
            [np.float32, (44, 6, 8, 4), 3, 0, 1],
            [np.float32, (4, 5, 6), 2, 0, 1],
            [np.float32, (4, 4, 2, 2, 6, 4), 2, 0, 1],
            [np.float32, (4, 2, 1, 5, 8, 10), 0, 0, 1],
            [np.float32, (4, 2, 1, 5, 8, 1, 2, 3), 0, 0, 1],
            [np.float32, (8, 10, 1, 5, 2, 10), 0, 0, 1],

            [np.float16, (12000, 10), 0, 0, 1],
            [np.float16, (6000, 20, 10), 0, 0, 1],
            [np.float16, (4, 6), -1, 100, 200],
            [np.float16, (2, 2, 3), 1, 100, 200],
            [np.float16, (4, 6, 8, 10), 3, 0, 1],
            [np.float16, (4, 5, 6), 2, 0, 1],
            [np.float16, (22, 3, 35, 34, 10, 2), 0, 1, 10],
            [np.float16, (42, 33, 32, 32, 36, 22), -3, 1, 10]
        ]
        for item in shape_format:
            input_data = self.generate_single_data(item[3], item[4], item[1], item[0])
            cpu_output = self.cpu_op_exec(input_data, item[2])
            npu_output = self.npu_op_exec(input_data, item[2])
            self.assertRtolEqual(cpu_output, npu_output, prec16=0.002, prec=0.0002)


if __name__ == "__main__":
    run_tests()
