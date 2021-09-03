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
import torch.nn.functional as F
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor

class TestUpsampleNearest2DBackward(TestCase):
    def cpu_op_exec(self, input, size):
        input.requires_grad_(True)
        output = F.interpolate(input, size, mode = "nearest")
        output.backward(torch.ones_like(output))
        return output.detach().numpy(), input.grad.numpy()
    
    def npu_op_exec(self, input, size):
        input.requires_grad_(True)
        output = F.interpolate(input, size, mode = "nearest")
        inputback = torch.ones_like(output)
        output.backward(inputback)
        out = output.to("cpu")
        grad = input.grad
        grad = grad.to("cpu")
        return out.detach().numpy(), grad.detach().numpy()

    def test_upsample_bilinear2d_shape_format(self, device):
        shape_format = [
                        [[np.float32, 0, (2, 3, 4, 4)], [2, 2]],
                        [[np.float16, 0, (2, 3, 4, 4)], [2, 2]],
                        [[np.float32, 0, (5, 3, 6, 4)], [10, 10]],
                        [[np.float16, 0, (5, 3, 6, 4)], [10, 10]],
                        [[np.float32, 0, (2, 3, 2, 4)], [10, 10]],
                        [[np.float16, -1, (2, 3, 2, 3)], [10, 10]]
                        ] 

        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            if cpu_input == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, item[1])
            npu_output, npu_grad = self.npu_op_exec(npu_input, item[1])

            cpu_grad = cpu_grad.astype(npu_grad.dtype)
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)

instantiate_device_type_tests(TestUpsampleNearest2DBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()