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
from util_test import create_common_tensor
from graph_utils import graph_mode

class TestUpsampleLinear1DBackward(TestCase):
    def cpu_op_exec(self, input, grads, size, align_corners):
        input.requires_grad_(True)
        output = torch._C._nn.upsample_linear1d(input, size, align_corners=align_corners)
        output.backward(grads)
        gradcpu = input.grad
        return output.detach().numpy(), gradcpu.detach().numpy()

    def npu_op_exec(self, input, grads, size, align_corners):
        input.requires_grad_(True)
        output = torch._C._nn.upsample_linear1d(input, size, align_corners=align_corners)
        output = output.to("npu")
        output.backward(grads)  
        gradnpu = input.grad
        gradnpu = gradnpu.to("cpu")
        output = output.to("cpu")
        return output.detach().numpy(), gradnpu.detach().numpy()

    @graph_mode
    def test_upsample_linear1d_backward_shape_format(self, device):
        test_cases = [
            [[np.float16, 0, (1, 1, 1, 2)], [4, ], True],
            [[np.float16, 0, (2, 1, 1, 4)], [8, ], True],
            [[np.float16, 0, (2, 2, 1, 3)], [1, ], True],
            [[np.float16, 0, (2, 1, 1, 1)], [4, ], False],
            [[np.float16, 0, (4, 1, 1, 2)], [4, ], False],
            [[np.float16, 0, (1, 1, 1, 1)], [1, ], False],

            [[np.float32, 0, (1, 1, 1, 2)], [4, ], True],
            [[np.float32, 0, (2, 1, 1, 2)], [4, ], True],
            [[np.float32, 0, (2, 2, 1, 3)], [1, ], True],
            [[np.float32, 0, (3, 1, 1, 1)], [2, ], False],
            [[np.float32, 0, (4, 1, 1, 1)], [2, ], False],
            [[np.float32, 0, (1, 1, 1, 1)], [1, ], False],

            [[np.float32, 0, (9, 7, 1, 2)], [15, ], True],
            [[np.float16, 0, (8, 7, 1, 1)], [2, ], True],
            [[np.float16, 0, (17, 2, 1, 3)], [1, ], True],
            [[np.float16, 0, (6, 4, 1, 1)], [3, ], False],
            [[np.float16, 0, (8, 7, 1, 2)], [4, ], False],
            [[np.float16, 0, (2, 7, 1, 7)], [1, ], False],

            [[np.float32, 0, (9, 7, 1, 2)], [7, ], True],
            [[np.float32, 0, (8, 3, 1, 1)], [2, ], True],
            [[np.float32, 0, (8, 3, 1, 1)], [2, ], True],
            [[np.float32, 0, (17, 2, 1, 3)], [1, ], True],
            [[np.float32, 0, (9, 7, 1, 2)], [7, ], False],
            [[np.float32, 0, (8, 3, 1, 3)], [2, ], False],
            [[np.float32, 0, (2, 7, 1, 7)], [1, ], False],

            [[np.float16, 0, (9, 7, 1, 2)], [17, ], True],
            [[np.float16, 0, (17, 13, 1, 15)], [16, ], True],
            [[np.float16, 0, (61, 41, 1, 1)], [7, ], False],
            [[np.float16, 0, (38, 7, 1, 7)], [16, ], False],
            [[np.float32, 0, (997, 3, 1, 1)], [32, ], True],
            [[np.float32, 0, (627, 2, 1, 3)], [17, ], False],
            [[np.float32, 0, (78, 73, 1, 1)], [48, ], False],
            [[np.float32, 0, (6553, 2, 1, 2)], [4, ], False],
            [[np.float16, 0, (6553, 2, 1, 2)], [4, ], False],
            [[np.float32, 0, (1008, 3, 1, 2)], [4, ], False],
            [[np.float16, 0, (1008, 3, 1, 2)], [4, ], False]
        ]
        for item in test_cases:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)

            size = list(item[0][2])
            size[3] = item[1][0]

            grad_item = []
            grad_item.append(item[0][0])
            grad_item.append(item[0][1])
            grad_item.append(size)
            cpu_grads, npu_grads = create_common_tensor(grad_item, 0, 100)

            if cpu_input.dtype == torch.float16:
                cpu_input = cpu_input.to(torch.float32)

            if cpu_grads.dtype == torch.float16:
                cpu_grads = cpu_grads.to(torch.float32)

            if cpu_input.dim() == 4:
                cpu_input = cpu_input.squeeze(2)

            if npu_input.dim() == 4:
                npu_input = npu_input.squeeze(2)

            if cpu_grads.dim() == 4:
                cpu_grads = cpu_grads.squeeze(2)

            if npu_grads.dim() == 4:
                npu_grads = npu_grads.squeeze(2)

            size = item[1]
            align_corners = item[2]

            cpu_output, cpu_grad = self.cpu_op_exec(cpu_input, cpu_grads, size, align_corners)
            npu_output, npu_grad = self.npu_op_exec(npu_input, npu_grads, size, align_corners)

            cpu_output = cpu_output.astype(npu_output.dtype)
            cpu_grad = cpu_grad.astype(npu_grad.dtype)

            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_grad, npu_grad)



instantiate_device_type_tests(TestUpsampleLinear1DBackward, globals(), except_for="cpu")
if __name__ == "__main__":
    run_tests()
