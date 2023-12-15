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

import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestSubSample(TestCase):
    def get_num(self, input1, output):
        input_num1 = 0
        input_num0 = 0
        output_num1 = 0
        output_num0 = 0
        for i in range(input1.size()[0]):
            if input1[i] == 1:
                input_num1 = input_num1 + 1
            if input1[i] == 0:
                input_num0 = input_num0 + 1
        for i in range(output.size()[0]):
            if output[i] == 1:
                output_num1 = output_num1 + 1
            if output[i] == 0:
                output_num0 = output_num0 + 1
        list1 = [input_num1, input_num0, output_num1, output_num0]

        return list1

    def numless_equal(self, input_num1, input_num0, output_num1, output_num0, size, fraction):
        error_name = "result error"
        if input_num1 < size * fraction:
            if output_num1 != input_num1:
                self.fail(error_name)
            if input_num0 < size - input_num1 and output_num0 != input_num0:
                self.fail(error_name)
            if input_num0 >= size - input_num1 and output_num0 != size - input_num1:
                self.fail(error_name)

    def nummore_equal(self, input_num1, input_num0, output_num1, output_num0, size, fraction):
        error_name = "result error"
        if input_num1 >= size * fraction:
            if output_num1 != size * fraction:
                self.fail(error_name)
            if input_num0 < size - size * fraction and output_num0 != input_num0:
                self.fail(error_name)
            if input_num0 >= size - size * fraction and output_num0 != size - size * fraction:
                self.fail(error_name)

    def test_subsample(self, device="npu"):
        for _ in range(20):
            input1 = np.random.randint(-1, 2, size=(10))
            npu_input = torch.from_numpy(input1).to("npu")
            # input only suport int32
            npu_input = npu_input.to(torch.int32)
            npu_output1 = torch_npu.npu_sub_sample(npu_input, 5, 0.6)
            getlist = self.get_num(npu_input, npu_output1)
            self.numless_equal(getlist[0], getlist[1], getlist[2], getlist[3], 5, 0.6)
            self.nummore_equal(getlist[0], getlist[1], getlist[2], getlist[3], 5, 0.6)


if __name__ == "__main__":
    run_tests()
