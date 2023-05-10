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
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.decorator import graph_mode


class TestUpsampleBicubic2dBackward(TestCase):

    def cpu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        input1.requires_grad = True
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.detach().numpy()
        return output_grad

    def npu_op_exec(self, input1, output_size, align_corners, scale_h, scale_w):
        input1.requires_grad = True
        output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu").detach().numpy()
        return output_grad

    @graph_mode
    def test_upsample_bicubic2d_common_shape_format(self):
        shape_format = [
            [[np.float32, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],
            [[np.float32, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (10, 10, 786432, 8)], (786432, 8), False, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), True, 0, 0, 0, 255],
            [[np.float32, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float32, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 3402823500.0]
        ]
        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)

    @graph_mode
    def test_upsample_bicubic2d_float16_shape_format(self):
        def cpu_op_exec_fp16(input1, output_size, align_corners, scale_h, scale_w):
            input1 = input1.to(torch.float32)
            input1.requires_grad = True
            output = torch._C._nn.upsample_bicubic2d(input1, output_size, align_corners, scale_h, scale_w)
            output.backward(torch.ones_like(output))
            output_grad = input1.grad
            output_grad = output_grad.detach().numpy()
            output_grad = output_grad.astype(np.float16)
            return output_grad

        shape_format = [
            [[np.float16, -1, (1, 1, 1, 1)], (1, 1), True, 0, 0, 0, 255],
            [[np.float16, -1, (2, 65535, 2, 2)], (2, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (32, 32, 32, 32)], (32, 32), False, 0, 0, 0, 6550.0],
            [[np.float16, -1, (1, 1, 1, 1)], (2, 2), True, 0, 0, 0, 255],
            [[np.float16, -1, (1, 1, 1, 1)], (2, 2), False, 0.5, 0.5, 0, 255],
            [[np.float16, -1, (1, 1, 2, 2)], (4, 4), False, 0.5, 0.5, 0, 255],
            [[np.float16, -1, (32, 32, 32, 32)], (64, 64), False, 0.5, 0.5, 0, 6550.0]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[5], item[6])
            cpu_output = cpu_op_exec_fp16(cpu_input1, item[1], item[2], item[3], item[4])
            npu_output = self.npu_op_exec(npu_input1, item[1], item[2], item[3], item[4])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
