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
import torch.nn.functional as F
import numpy as np
from common_utils import TestCase, run_tests
from common_device_type import instantiate_device_type_tests
from util_test import create_common_tensor

class TestMishBackward(TestCase):
    def npu_op_exec(self, input1):
        input1.requires_grad = True
        output = torch.npu_mish(input1)
        output.backward(torch.ones_like(output))
        output_grad = input1.grad
        output_grad = output_grad.to("cpu")
        output_grad = output_grad.detach().numpy()
        output = output.cpu().detach().numpy()
        return output_grad, output

    def test_mish_fp32(self, device):
        npu_input = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]).npu()
        ep_output_grad = torch.tensor([1.0490363, 1.0693179, 1.021107, 1.0044329, 1.0008003, 1.0001341, 1.0000216, 1.0000033, 1.0000005, 1.0000001])
        ep_npu_output = torch.tensor([0.8652344, 1.9439697, 2.9865417, 3.9974136, 4.999552, 5.9999266, 6.9999886, 7.999998, 8.999999, 10.])
        output_grad, npu_output = self.npu_op_exec(npu_input)
        self.assertRtolEqual(ep_output_grad.numpy(), output_grad)
        self.assertRtolEqual(ep_npu_output.numpy(), npu_output)

instantiate_device_type_tests(TestMishBackward, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
