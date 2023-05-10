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


class TestBatchMatMulV2(TestCase):

    def supported_op_exec(self, input1, input2, output_sizes):
        output = torch.matmul(input1, input2)
        return output.cpu().detach()

    def custom_op_exec(self, input1, input2, output_sizes):
        output = torch_npu.npu_bmmV2(input1, input2, output_sizes)
        return output.cpu().detach()

    def test_npu_bmmV2(self, device="npu"):
        item1 = [np.float32, 0, (10, 3, 4)]
        item2 = [np.float32, 0, (10, 4, 5)]
        _, npu_input1 = create_common_tensor(item1, -1, 1)
        _, npu_input2 = create_common_tensor(item2, -1, 1)
        output_sizes = []

        supported_output = self.supported_op_exec(npu_input1, npu_input2, output_sizes)
        custom_output = self.custom_op_exec(npu_input1, npu_input2, output_sizes)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
