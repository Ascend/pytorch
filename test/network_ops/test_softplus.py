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
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestSoftplus(TestCase):
    def cpu_op_exec(self, input1, beta1, threshold1):
        softplus = torch.nn.Softplus(beta=beta1, threshold=threshold1)
        output = softplus(input1)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, beta1, threshold1):
        softplus = torch.nn.Softplus(beta=beta1, threshold=threshold1)
        output = softplus(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def cpu_default_op_exec(self, input1):
        softplus = torch.nn.Softplus()
        output = softplus(input1)
        output = output.numpy()
        return output

    def npu_default_op_exec(self, input1):
        softplus = torch.nn.Softplus()
        output = softplus(input1)
        output = output.to("cpu")
        output = output.numpy()
        return output

    def test_softplus_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (2, 3)]],
            [[np.float32, 0, (8, 4, 3, 9)], 0.5, 5.],
            [[np.float32, 0, (1, 7)], 0.5, 20.],
            [[np.float32, 0, (1, 5, 6)], 1., 5.],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1, item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1)
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)

    def test_softplus_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (2, 3)]],
            [[np.float16, 0, (8, 4, 3, 9)], 0.5, 5.],
            [[np.float16, 0, (1, 7)], 0.5, 20.],
            [[np.float16, 0, (1, 5, 6)], 1., 5.],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -2, 100)
            if len(item) > 1:
                cpu_output = self.cpu_op_exec(cpu_input1.to(torch.float32), item[1], item[2])
                npu_output = self.npu_op_exec(npu_input1, item[1], item[2])
            else:
                cpu_output = self.cpu_default_op_exec(cpu_input1.to(torch.float32))
                npu_output = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)


if __name__ == "__main__":
    run_tests()
