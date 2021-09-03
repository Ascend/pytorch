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

import time

import torch
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestBroadcastToContiguous(TestCase):
    def test_BroadcastToContiguous(self, device):
        def cpu_op_exec(input1, shape):
            output = input1.expand(shape).contiguous()
            output = output.numpy()
            return output

        def npu_op_exec(input1, shape):
            output = input1.expand(shape).contiguous()
            output = output.to("cpu")
            output = output.numpy()
            return output

        shape_format = [
                        [[np.float32, 0, (1, 4)], [3, 4]],
                        [[np.float16, 0, (5, 1, 6, 1)], [5, 2, 6, 4]],
                       ]
        start = time.time()
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], 0, 100)
            cpu_output = cpu_op_exec(cpu_input1, item[1])
            npu_output = npu_op_exec(npu_input1, item[1])
            self.assertRtolEqual(cpu_output, npu_output)
        end = time.time()
        print("Broadcast to contiguous uses: %.2f s"%(end-start))      

instantiate_device_type_tests(TestBroadcastToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
