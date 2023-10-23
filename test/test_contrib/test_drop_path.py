# Copyright (c) 2020 Huawei Technologies Co., Ltd
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

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import NpuDropPath


class TestDropPath(TestCase):
    def npu_slow_drop_path_op_exec(self, input1, input2):
        class DropPath(nn.Module):
            def __init__(self, drop_prob=None):
                super(DropPath, self).__init__()
                self.drop_prob = drop_prob

            def forward(self, x):
                return drop_path(x, self.drop_prob, self.training)

        def drop_path(x, drop_prob: float = 0., training: bool = False):
            if math.isclose(drop_prob, 0.) or not training:
                return x

            keep_prob = 1 - drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + \
                torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            output = x.div(keep_prob) * random_tensor
            return output

        slow_drop_path = DropPath(0).npu()
        output = input1 + slow_drop_path(input2)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_drop_path_op_exec(self, input1, input2):
        fast_drop_path = NpuDropPath(0).npu()
        output = input1 + fast_drop_path(input2)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_drop_path(self, input1, input2):
        output = self.npu_slow_drop_path_op_exec(input1, input2)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_drop_path_op_exec(input1, input2)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_drop_path(self, input1, input2):
        output = self.npu_fast_drop_path_op_exec(input1, input2)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_drop_path_op_exec(input1, input2)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    def test_drop_path_shape_format(self):
        shape_format = [
            [[np.float16, 2, [50, 25, 7, 100]], [np.float16, 2, [50, 25, 7, 100]]],
            [[np.float16, 3, [68, 5, 75, 16]], [np.float16, 3, [68, 5, 75, 16]]],
            [[np.float16, 3, [68, 5]], [np.float16, 3, [68, 5]]],
            [[np.float32, 2, [50, 25, 7, 100]], [np.float32, 2, [50, 25, 7, 100]]],
            [[np.float32, 3, [68, 5, 75, 16]], [np.float32, 3, [68, 5, 75, 16]]],
            [[np.float32, 3, [68, 5]], [np.float32, 3, [68, 5]]],
        ]

        for item in shape_format:
            _, mat1_npu = create_common_tensor(item[0], -10, 10)
            _, mat2_npu = create_common_tensor(item[1], -10, 10)
            mat1_npu.requires_grad_(True)
            mat2_npu.requires_grad_(True)
            slow_output, slow_time = \
                self.npu_slow_drop_path(mat1_npu, mat2_npu)
            fast_output, fast_time = \
                self.npu_fast_drop_path(mat1_npu, mat2_npu)

            self.assertRtolEqual(slow_output, fast_output)
            self.assertTrue(slow_time > fast_time)


if __name__ == "__main__":
    run_tests()
