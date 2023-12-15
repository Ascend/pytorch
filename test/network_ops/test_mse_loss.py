# Copyright (c) 2020 Huawei Technologies Co., Ltd
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
from torch_npu.testing.common_utils import create_common_tensor


class TestMseLoss(TestCase):

    def generate_mse_inputs(self, min_ref, max_ref, shape, dtype):
        input1 = np.random.uniform(min_ref, max_ref, shape).astype(dtype)
        input2 = np.random.uniform(min_ref, max_ref, shape).astype(dtype)

        npu_input1 = torch.from_numpy(input1)
        npu_input2 = torch.from_numpy(input2)

        return npu_input1, npu_input2

    def cpu_op_exec(self, input1, input2, reduction):
        if reduction == "":
            output = torch.nn.functional.mse_loss(input1, input2)
        else:
            output = torch.nn.functional.mse_loss(input1, input2, reduction=reduction)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, input2, reduction):
        input1 = input1.to("npu")
        input2 = input2.to("npu")
        if reduction == "":
            output = torch.nn.functional.mse_loss(input1, input2)
        else:
            output = torch.nn.functional.mse_loss(input1, input2, reduction=reduction)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_mse_loss_shape_format(self, device="npu"):
        shape_format = [
            [0, 100, (4, 3), np.float32, ""],
            [0, 100, (4, 3), np.float32, "mean"],
            [0, 100, (4, 3), np.float32, "none"],
            [0, 100, (4, 3), np.float32, "sum"],
        ]
        for item in shape_format:
            print("test ", item[3], item[4])
            npu_input1, npu_input2 = self.generate_mse_inputs(item[0], item[1], item[2], item[3])
            cpu_output = self.cpu_op_exec(npu_input1, npu_input2, item[4])
            npu_output = self.npu_op_exec(npu_input1, npu_input2, item[4])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_mse_mix_dtype(self, device="npu"):
        npu_input1, npu_input2 = create_common_tensor([np.int32, 0, (2, 3)], 1, 100)
        npu_input3, npu_input4 = create_common_tensor([np.float32, 0, (2, 3)], 1, 100)
        cpu_output = self.cpu_op_exec(npu_input1, npu_input3, "mean")
        npu_output = self.npu_op_exec(npu_input1, npu_input3, "mean")
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
