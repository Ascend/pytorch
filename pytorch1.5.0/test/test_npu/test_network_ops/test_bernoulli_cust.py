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
import copy
import sys
import numpy as np
import torch
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestBernoulli(TestCase):

    def npu_op_inplace_float_exec(self, input0):
        output = input0.bernoulli_cust_(0.5)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_bernoulli_float_p(self, device):
        format_list = [0, 2]
        shape_list = [(128, 256)]
        shape_format = [
            [np.float32, i, j] for i in format_list for j in shape_list
        ]
        for item in shape_format:
            cpu_input, npu_input = create_common_tensor(item, 0, 1)
            npu_output = self.npu_op_inplace_float_exec(npu_input)
            print(npu_output)

instantiate_device_type_tests(TestBernoulli, globals(), except_for="cpu")
if __name__ == '__main__':
    run_tests()

