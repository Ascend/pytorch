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


class TestSoftmaxCrossEntropyWithLogits(TestCase):

    def supported_op_exec(self, input1, label):
        softmax = torch.nn.functional.softmax(input1)
        log_softmax = torch.log(softmax)
        loss = torch.sum(- label * log_softmax, dim=1)
        return loss.cpu().detach()

    def custom_op_exec(self, input1, label):
        output = torch_npu.npu_softmax_cross_entropy_with_logits(input1, label)
        return output.cpu().detach()

    def test_npu_softmax_cross_entropy_with_logits(self, device="npu"):
        item = [np.float32, 0, (64, 10)]
        _, npu_input = create_common_tensor(item, -1, 1)
        _, label = create_common_tensor(item, 0, 1)
        supported_output = self.supported_op_exec(npu_input, label)
        custom_output = self.custom_op_exec(npu_input, label)
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
