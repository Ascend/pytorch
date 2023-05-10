# Copyright (c) 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestConfusionTranspose(TestCase):

    def supported_op_exec(self, input1, perm, shape, transpose_first):
        if transpose_first:
            output = input1.permute(*perm).contiguous().view(shape)
        else:
            output = input1.view(shape).permute(*perm)
        return output.cpu().detach()

    def custom_op_exec(self, input1, perm, shape, transpose_first):
        output = torch_npu.npu_confusion_transpose(input1, perm, shape, transpose_first)
        return output.cpu().detach()

    def test_npu_confusion_transpose(self, device="npu"):
        item = [np.float32, 0, [1, 576, 2560]]
        _, npu_input = create_common_tensor(item, 0, 100)
        perm = (0, 2, 1, 3)
        shape = [1, 576, 32, 80]
        transpose_first = False

        supported_output = self.supported_op_exec(npu_input, perm, shape, transpose_first)
        custom_output = self.custom_op_exec(npu_input, perm, shape, transpose_first)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
