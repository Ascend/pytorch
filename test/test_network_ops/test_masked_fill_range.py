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

import itertools
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestMaskedFillRange(TestCase):
    def cpu_op_exec(self, input1, start, end, value, axis, dim):
        out = input1.clone()
        start_shape = start.shape
        iter_list = itertools.product(list(range(start_shape[0])), list(range(start_shape[1])))

        def fill_each_pos(i, j, k, dim, axis, out, value):
            if dim == 1:
                out[k] = value[i]
            elif dim == 2:
                if axis == 0:
                    out[k, :] = value[i]
                else:
                    out[j, k] = value[i]
            elif dim == 3:
                if axis == 0:
                    out[k, :, :] = value[i]
                elif axis == 1:
                    out[:, k, :] = value[i]
                else:
                    out[j, :, k] = value[i]

        for i, j in iter_list:
            for k in range(start[i, j], end[i, j]):
                fill_each_pos(i, j, k, dim, axis, out, value)

        return out

    def npu_op_exec(self, input1, start, end, value, axis):
        out = torch_npu.npu_masked_fill_range(input1, start, end, value, axis)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_normalize_batch(self):
        shape_format = [
            [[np.float32, -1, [32, 64, 1688]],
             [list(range(0, 32))],
             [list(range(6, 38))], [[1], torch.float32], 2],
            [[np.float16, -1, [32, 64, 1688]],
             [list(range(0, 32))],
             [list(range(6, 38))], [[1], torch.float16], 2],
            [[np.int32, -1, [32, 64, 1688]],
             [list(range(0, 32))],
             [list(range(6, 38))], [[1], torch.int32], 2],
            [[np.int8, -1, [32, 64, 1688]],
             [list(range(0, 32))],
             [list(range(6, 38))], [[1], torch.int8], 2],
        ]
        for item in shape_format:
            axis = item[-1]
            cpu_input1, npu_input1 = create_common_tensor(item[0], 1, 100)
            shape = item[0][-1]
            cpu_start = torch.tensor(item[1], dtype=torch.int32)
            npu_start = cpu_start.npu()
            cpu_end = torch.tensor(item[2], dtype=torch.int32)
            npu_end = cpu_end.npu()
            cpu_value = torch.tensor(item[3][0], dtype=item[3][1])
            npu_value = cpu_value.npu()
            cpu_output = self.cpu_op_exec(cpu_input1, cpu_start, cpu_end, cpu_value, axis, len(shape))
            npu_output = self.npu_op_exec(npu_input1, npu_start, npu_end, npu_value, axis)
            cpu_output = cpu_output.numpy()
            cpu_output = cpu_output.astype(npu_output.dtype)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
