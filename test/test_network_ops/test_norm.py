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
from torch_npu.testing.common_utils import create_common_tensor


class TestNorm(TestCase):
    def norm_output_size(self, data, dimVal, keepdimVal):
        output_size = list(data.size())
        count = 0
        for i in dimVal:
            if i < 0:
                i = i + data.dim()
            if i < data.dim() and keepdimVal:
                output_size[i] = 1
            if i < data.dim() and not keepdimVal:
                output_size.pop(i - count)
                count += 1
        return output_size

    def cpu_dtype_out_exec(self, data, pVal, dimVal, keepdimVal, dtypeVal):
        output_size = self.norm_output_size(data, dimVal, keepdimVal)
        cpu_output = torch.randn(output_size)
        torch.norm(data, p=pVal, dim=dimVal, keepdim=keepdimVal, out=cpu_output, dtype=dtypeVal)
        return cpu_output.numpy()

    def npu_dtype_out_exec(self, data, pVal, dimVal, keepdimVal, dtypeVal):
        output_size = self.norm_output_size(data, dimVal, keepdimVal)
        npu_output = torch.randn(output_size).npu()
        torch.norm(data, p=pVal, dim=dimVal, keepdim=keepdimVal, out=npu_output, dtype=dtypeVal)
        return npu_output.cpu().numpy()

    def dtype_out_test(self, item):
        cpu_input, npu_input = create_common_tensor(item[0], 0, 100)
        cpu_out = self.cpu_dtype_out_exec(cpu_input, 2.0, [1, 2], True, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 2.0, [1, 2], True, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 2.5, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 2.5, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 1.0, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 1.0, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, 3.0, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, 3.0, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, -1.5, [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, -1.5, [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

        cpu_out = self.cpu_dtype_out_exec(cpu_input, float("-inf"), [1, 2], False, torch.float)
        npu_out = self.npu_dtype_out_exec(npu_input, float("-inf"), [1, 2], False, torch.float)
        self.assertRtolEqual(cpu_out, npu_out)

    def test_norm_shape_format(self):
        shape_format = [
            [[np.float32, 0, (64, 64, 64, 64)]],
            [[np.float32, 0, (128, 64, 30)]],
        ]

        for item in shape_format:
            # norm.dtype_out
            self.dtype_out_test(item)


if __name__ == "__main__":
    run_tests()
