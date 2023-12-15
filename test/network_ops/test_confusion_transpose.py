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

import copy
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConfusionTransposeD(TestCase):
    def npu_op_exec(self, input1, shape, perm, transpose_first):
        output = torch_npu.npu_confusion_transpose(input1, perm, shape, transpose_first)
        output = output.cpu().numpy()
        return output

    def cpu_op_exec(self, input1, shape, perm, transpose_first):
        if transpose_first:
            output = input1.permute(*perm).contiguous().view(shape)
        else:
            output = input1.view(shape).permute(*perm)
        output = output.numpy()
        return output

    def test_confusion_transpose(self, device="npu"):
        shape_format = [
            [[np.float32, 0, [1, 576, 2560]], [1, 576, 32, 80], (0, 2, 1, 3), False],
            [[np.float32, 0, [1, 32, 576, 80]], [1, 576, 2560], (0, 2, 1, 3), True],
            [[np.float16, 0, [1, 576, 2560]], [1, 576, 32, 80], (0, 2, 1, 3), False],
            [[np.float16, 0, [1, 32, 576, 80]], [1, 576, 2560], (0, 2, 1, 3), True],
            [[np.int_, 0, [1, 576, 2560]], [1, 576, 32, 80], (0, 2, 1, 3), False],
            [[np.int_, 0, [1, 32, 576, 80]], [1, 576, 2560], (0, 2, 1, 3), True],
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
            cpu_output = self.cpu_op_exec(cpu_input, item[1], item[2], item[3])
            npu_output = self.npu_op_exec(npu_input, item[1], item[2], item[3])
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
