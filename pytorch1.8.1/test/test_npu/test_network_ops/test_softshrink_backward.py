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
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests

cpu_input_grad = None
npu_input_grad = None

def cpu_input_grad_hook(grad):
    global cpu_input_grad
    cpu_input_grad = grad.numpy()

def npu_input_grad_hook(grad):
    global npu_input_grad
    npu_input_grad = grad.cpu().numpy()

class TestSoftShrinkBackward(TestCase):
    def generate_data(self, min_d, max_d, shape, dtype):
        input_grad = np.random.uniform(min_d, max_d, shape).astype(dtype)
        input_x = np.random.uniform(min_d, max_d, shape).astype(dtype)
        npu_input_grad_data = torch.from_numpy(input_grad)
        npu_input_x = torch.from_numpy(input_x)
        return npu_input_grad_data, npu_input_x

    def cpu_op_exec(self, input_x, input_grad, lambd):
        input_x.requires_grad_(True)
        input_x.register_hook(cpu_input_grad_hook)
        m = torch.nn.Softshrink(lambd=lambd)
        output = m(input_x)
        output.backward(input_grad)

    def npu_op_exec(self, input_x, input_grad, lambd):
        input_x = input_x.to("npu")
        input_grad = input_grad.to("npu")
        input_x.requires_grad_(True)
        input_x.register_hook(npu_input_grad_hook)
        m = torch.nn.Softshrink(lambd=lambd).npu()
        output = m(input_x)
        output.backward(input_grad)

    def test_softshrink_3_3_float32(self, device):
        input_grad, input_x = self.generate_data(-1, 1, (3, 3), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_softshrink_100_100_float32(self, device):
        input_grad, input_x = self.generate_data(-1, 1, (100, 100), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

    def test_softshrink_10_10_10_10_float32(self, device):
        input_grad, input_x = self.generate_data(-1, 1, (10, 10, 10, 10), np.float32)
        self.cpu_op_exec(input_x, input_grad, 0.5)
        self.npu_op_exec(input_x, input_grad, 0.5)
        self.assertRtolEqual(cpu_input_grad, npu_input_grad)

instantiate_device_type_tests(TestSoftShrinkBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()