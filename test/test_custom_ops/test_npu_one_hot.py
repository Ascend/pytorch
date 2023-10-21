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

import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuOneHot(TestCase):

    def custom_one_hot(self, input1, num_classes=-1, depth=1, on_value=1, off_value=0):
        output = torch.nn.functional.one_hot(input1, num_classes).float()
        output[output == 1] = on_value
        output[output == 0] = off_value
        output_dim = output.dim()
        if output.size(-1) >= depth:
            output = torch.index_select(
                input=output,
                dim=output_dim - 1,
                index=torch.IntTensor(range(depth)).npu()
            )
        else:
            output_size = list(output.size())
            output_size[-1] = depth - output_size[-1]
            zero_size = output_size
            zero = torch.zeros(size=zero_size).npu()
            output = torch.cat((output, zero), dim=output_dim - 1)
        return output

    def custom_op_exec(self, input1, num_classes=-1, depth=1, on_value=1, off_value=0):
        output = self.custom_one_hot(
            input1, num_classes, depth, on_value, off_value)
        output = output.cpu().numpy()
        return output

    def npu_op_exec(self, input1, num_classes, depth=1, on_value=1, off_value=0):
        output = torch_npu.npu_one_hot(
            input1, num_classes, depth, on_value, off_value)
        output = output.cpu().numpy()
        return output

    def test_one_hot(self):
        input1 = [
            torch.randint(1, 4, (2, 3)).npu(),
            torch.randint(1, 4, (1, 2, 3)).npu(),
            torch.randint(1, 4, (2, 2, 3)).npu(),
        ]

        for item in input1:
            custom_output = self.custom_op_exec(item, -1, 3, 1, 0)
            npu_output = self.npu_op_exec(item, -1, 3, 1, 0)
            self.assertRtolEqual(custom_output, npu_output)

        for item in input1:
            custom_output = self.custom_op_exec(item, -1, 10, 1, 0)
            npu_output = self.npu_op_exec(item, -1, 10, 1, 0)
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
