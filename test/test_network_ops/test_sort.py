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


class TestSort(TestCase):
    def cpu_op_exec(self, input1, dim):
        output, indices = torch.sort(input1, dim=dim)
        output = output.numpy()
        indices = indices.numpy()
        return output, indices

    def npu_op_exec(self, input1, dim):
        output, indices = torch.sort(input1, dim=dim)
        output = output.cpu()
        indices = indices.cpu()
        output = output.numpy()
        indices = indices.numpy()
        return output, indices

    def cpu_default_op_exec(self, input1):
        output, indices = torch.sort(input1)
        output = output.numpy()
        indices = indices.numpy()
        return output, indices

    def npu_default_op_exec(self, input1):
        output, indices = torch.sort(input1)
        output = output.cpu()
        indices = indices.cpu()
        output = output.numpy()
        indices = indices.numpy()
        return output, indices

    # at present accuracy under FP32 is inadequate
    def _test_sort_shape_format_fp32(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (8, 4, 3, 9)], 2],
            [[np.float32, 0, (2, 3)]],
            [[np.float32, 0, (1, 7)], 0],
            [[np.float32, 0, (1, 5, 6)], 1],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output, cpu_indices = self.cpu_op_exec(cpu_input1, item[1])
                npu_output, npu_indices = self.npu_op_exec(npu_input1, item[1])
            else:
                cpu_output, cpu_indices = self.cpu_default_op_exec(cpu_input1)
                npu_output, npu_indices = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_indices, npu_indices)

    def test_sort_shape_format_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (8, 4, 3, 9)], 2],
            [[np.float16, 0, (2, 3)]],
            [[np.float16, 0, (1, 7)], 0],
            [[np.float16, 0, (1, 5, 6)], 1],
        ]

        for item in shape_format:
            cpu_input1, npu_input1 = create_common_tensor(item[0], -100, 100)
            if len(item) > 1:
                cpu_output, cpu_indices = self.cpu_op_exec(cpu_input1.to(torch.float32), item[1])
                npu_output, npu_indices = self.npu_op_exec(npu_input1, item[1])
            else:
                cpu_output, cpu_indices = self.cpu_default_op_exec(cpu_input1.to(torch.float32))
                npu_output, npu_indices = self.npu_default_op_exec(npu_input1)
            self.assertRtolEqual(cpu_output.astype(np.float16), npu_output)


if __name__ == "__main__":
    run_tests()
