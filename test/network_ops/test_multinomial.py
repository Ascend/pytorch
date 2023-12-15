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

# coding: utf-8


import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMultinomial(TestCase):

    def sample_1d(self, weight, num_samples):
        for replacement in [True, False]:
            sample = torch.multinomial(weight, num_samples, replacement)
            for index in sample:
                self.assertNotEqual(weight[index], 0)

    def test_multinomial_1d_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (5,)], 0, 100, 5],
            [[np.float32, 0, (10,)], 0, 100, 10],
            [[np.float32, 0, (20,)], 0, 100, 10],
            [[np.float32, 0, (50,)], 0, 100, 5],
            [[np.float16, 0, (5,)], 0, 100, 5],
            [[np.float16, 0, (10,)], 0, 100, 10],
            [[np.float16, 0, (20,)], 0, 100, 10],
            [[np.float16, 0, (50,)], 0, 100, 5]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            self.sample_1d(npu_input1, item[3])

    def sample_2d(self, weight, num_samples):
        for replacement in [True, False]:
            sample = torch.multinomial(weight, num_samples, replacement)
            for i, row in enumerate(sample):
                for j in row:
                    self.assertNotEqual(weight[i][j], 0)

    def test_multinomial_2d_shape_format(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (5, 5)], 0, 100, 5],
            [[np.float32, 0, (5, 10)], 0, 100, 10],
            [[np.float32, 0, (5, 20)], 0, 100, 10],
            [[np.float32, 0, (5, 50)], 0, 100, 5],
            [[np.float16, 0, (5, 5)], 0, 100, 5],
            [[np.float16, 0, (5, 10)], 0, 100, 10],
            [[np.float16, 0, (5, 20)], 0, 100, 10],
            [[np.float16, 0, (5, 50)], 0, 100, 5]
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            self.sample_2d(npu_input1, item[3])

    def sample_seed(self, weight, num_samples):
        for replacement in [True, False]:
            torch.manual_seed(123)
            out1 = torch.multinomial(weight, num_samples, replacement)
            torch.manual_seed(123)
            out2 = torch.multinomial(weight, num_samples, replacement)
            self.assertRtolEqual(out1.cpu(), out2.cpu())

    def test_multinomial_seed(self):
        shape_format = [
            [[np.float32, 0, 20], 0, 100, 10],
            [[np.float32, 0, 50], 0, 100, 5],
            [[np.float16, 0, 5], 0, 100, 5],
            [[np.float16, 0, 10], 0, 100, 10],
            [[np.float32, 0, (5, 20)], 0, 100, 10],
            [[np.float32, 0, (5, 50)], 0, 100, 5],
            [[np.float16, 0, (5, 5)], 0, 100, 5],
            [[np.float16, 0, (5, 10)], 0, 100, 10],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], item[1], item[2])
            self.sample_seed(npu_input1, item[3])


if __name__ == "__main__":
    run_tests()
