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

import numpy as np
import torch
import torch.nn.functional as F

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuPad(TestCase):

    def custom_pad(self, input_data, pads):
        new_pads = pads[2:] + pads[:2]
        output = F.pad(input_data, new_pads, "constant", 0)
        return output

    def custom_op_exec(self, input_data, pads):
        output = self.custom_pad(input_data, pads)
        return output.cpu().numpy()

    def npu_op_exec(self, input_data, pads):
        output = torch_npu.npu_pad(input_data, pads)
        return output.cpu().numpy()

    def test_npu_pad(self):
        npu_input = torch.randn(2, 3).npu()
        pads_list = [(1, 1, 1, 1), (1, 2, 3, 4)]
        for pads in pads_list:
            custom_output = self.custom_op_exec(npu_input, pads)
            npu_output = self.npu_op_exec(npu_input, pads)
            self.assertRtolEqual(custom_output, npu_output)


if __name__ == "__main__":
    run_tests()
