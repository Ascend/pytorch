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


class TestPooling(TestCase):
    def generate_single_data(self, min_val, max_val, shape, dtype):
        input1 = np.random.uniform(min_val, max_val, shape).astype(dtype)
        npu_input1 = torch.from_numpy(input1)
        return npu_input1

    def cpu_op_exec(self, input_data):
        output = torch.max_pool2d(input_data, 2)
        return output

    def npu_op_exec(self, input_data):
        input_npu = input_data.to('npu')
        output = torch.max_pool2d(input_npu, 2)
        output = output.to("cpu")
        return output

    def test_maxpool_float16(self, device='npu'):
        params = [
            [-100, 10, (200, 10, 100, 60), np.float16],
            [-100, 50, (20, 10, 5, 10), np.float16]
        ]
        for para in params:
            input_data = self.generate_single_data(*para)
            input_data = input_data.to(torch.float32)
            cpu_output = self.cpu_op_exec(input_data)
            npu_output = self.npu_op_exec(input_data)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
