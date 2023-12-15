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


class TestUnique(TestCase):

    def test_unique(self):
        shape_format = [
            [[np.uint8, (2, 3)], True, True],
            [[np.int8, (2, 3)], True, True],
            [[np.int16, (2, 3)], True, True],
            [[np.int32, (2, 3)], True, True],
            [[np.long, (2, 3)], True, True],
            [[np.long, (5, 3)], True, False],
            [[np.long, (2, 3, 4)], False, False],
            [[np.long, (3, 3)], False, True],
            [[np.float32, (2, 3)], True, False],
            [[np.bool, (2, 3)], True, True],
            [[np.float16, (2, 3)], True, True],
            [[np.float16, (208, 3136, 19, 5)], False, False]
        ]

        for item in shape_format:
            input1 = np.random.uniform(-10, 10, item[0][1]).astype(item[0][0])
            cpu_input1 = torch.from_numpy(input1)
            if item[0][0] == np.float16:
                cpu_input1 = torch.from_numpy(input1.astype(np.float32))
            npu_input1 = torch.from_numpy(input1).npu()

            cpu_output_y, cpu_yInverse = torch._unique(cpu_input1, item[1], item[2])
            npu_output_y, npu_yInverse = torch._unique(npu_input1, item[1], item[2])

            cpu_output_y = cpu_output_y.numpy()
            if item[0][0] == np.float16:
                cpu_output_y = cpu_output_y.astype(np.float16)
            self.assertRtolEqual(cpu_output_y, npu_output_y.cpu().numpy())
            self.assertRtolEqual(cpu_yInverse.numpy(), npu_yInverse.cpu().numpy())


if __name__ == "__main__":
    run_tests()
