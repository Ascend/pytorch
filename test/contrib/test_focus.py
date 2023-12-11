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

import unittest
import time
import numpy as np
import torch
import torch.nn as nn
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.contrib.module import Focus
from torch_npu.contrib.module.focus import fast_slice


class TestFocus(TestCase):
    def npu_slow_focus_op_exec(self, c1, c2, input1):
        class Conv(nn.Module):
            def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
                super(Conv, self).__init__()
                self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
                self.bn = nn.BatchNorm2d(c2)
                self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

            def forward(self, x):
                return self.act(self.bn(self.conv(x)))

            def fuseforward(self, x):
                return self.act(self.conv(x))

        class SrcFocus(nn.Module):
            def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
                super(SrcFocus, self).__init__()
                self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

            def forward(self, x):
                return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                                            x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

        def autopad(k, p=None):
            if p is None:
                p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
            return p

        slow_focus = SrcFocus(c1, c2).npu()
        output = slow_focus(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_fast_focus_op_exec(self, c1, c2, input1):
        fast_focus = Focus(c1, c2).npu()
        output = fast_focus(input1)
        output.sum().backward()
        return output.cpu().detach().numpy()

    def npu_slow_focus(self, c1, c2, input1):
        output = self.npu_slow_focus_op_exec(c1, c2, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t1 = time.time()
        for _ in range(repeat_time):
            self.npu_slow_focus_op_exec(c1, c2, input1)
        torch.npu.synchronize()
        slow_time = (time.time() - t1) / repeat_time * 1000

        return output, slow_time

    def npu_fast_focus(self, c1, c2, input1):
        output = self.npu_fast_focus_op_exec(c1, c2, input1)

        repeat_time = 100
        torch.npu.synchronize()
        t2 = time.time()
        for _ in range(repeat_time):
            self.npu_fast_focus_op_exec(c1, c2, input1)
        torch.npu.synchronize()
        fast_time = (time.time() - t2) / repeat_time * 1000

        return output, fast_time

    def npu_slow_slice(self, input1):
        output = [input1[..., ::2, ::2], input1[..., 1::2, ::2],
                  input1[..., ::2, 1::2], input1[..., 1::2, 1::2]]

        return output

    def npu_fast_slice(self, input1):
        output = fast_slice(input1)

        return output

    def test_slice_shape_format(self):
        shape_format = [
            [np.float16, 2, [2, 3, 4, 5]],
            [np.float32, 2, [3, 5, 8, 9]],
        ]
        for item in shape_format:
            _, input1 = create_common_tensor(item, -10, 10)
            slow_output = self.npu_slow_slice(input1)
            fast_output = self.npu_fast_slice(input1)
            for i, _ in enumerate(slow_output):
                self.assertRtolEqual(slow_output[i].cpu().numpy(), fast_output[i].cpu().numpy())

    @unittest.skip("skip test_focus_shape_format now")
    def test_focus_shape_format(self):
        shape_format = [
            [[np.float16, 2, [20, 16, 50, 100]], 16, 33],
            [[np.float16, 2, [4, 8, 300, 40]], 8, 13],
        ]
        for item in shape_format:
            _, input1 = create_common_tensor(item[0], -10, 10)
            input1.requires_grad_(True)
            slow_output, slow_time = \
                self.npu_slow_focus(item[1], item[2], input1)
            fast_output, fast_time = \
                self.npu_fast_focus(item[1], item[2], input1)

            self.assertTrue(slow_time > fast_time)


if __name__ == "__main__":
    run_tests()
