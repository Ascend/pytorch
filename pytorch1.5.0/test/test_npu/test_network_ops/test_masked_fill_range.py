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
from torch.cuda import device
import torch
import numpy as np
import copy
import sys
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMaskedFillRange(TestCase):
    def cpu_op_exec(self, input1, start, end, value, axis, dim):
        out = input1.clone()
        start_shape = start.shape
        if dim == 1:
            for i in range(0, start_shape[0]):
                for j in range(0, start_shape[1]):
                    for k in range(start[i, j], end[i, j]):
                        out[k] = value[i]
        if dim == 2:
            for i in range(0, start_shape[0]):
                for j in range(0, start_shape[1]):
                    for k in range(start[i, j], end[i, j]):
                        if axis == 0:
                            out[k, :] = value[i]
                        else:
                            out[j, k] = value[i]
        if dim == 3:
            for i in range(0, start_shape[0]):
                for j in range(0, start_shape[1]):
                    for k in range(start[i, j], end[i, j]):
                        if axis == 0:
                            out[k, :, :] = value[i]
                        elif axis == 1:
                            out[:, k, :] = value[i]
                        else:
                            out[j, :, k] = value[i]
        return out

    def npu_op_exec(self, input1, start, end, value, axis):
        out = torch.npu_masked_fill_range(input1, start, end, value, axis)
        out = out.to("cpu")
        return out.detach().numpy()

    def test_normalize_batch(self, device):
        # TODO(ascend): 该算子还存在泛化问题， 目前保证模型场景没问题
        # Note: 以下为模型用例：测试通过
        shape_format = [
            [[np.float32, -1, [32, 64, 1688]], 
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                [[6, 7, 31, 9, 10, 11, 12, 19, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], [[1], torch.float32], 2],
            [[np.float16, -1, [32, 64, 1688]], 
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                [[6, 7, 31, 9, 10, 11, 12, 19, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], [[1], torch.float16], 2],
            [[np.int32, -1, [32, 64, 1688]], 
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                [[6, 7, 31, 9, 10, 11, 12, 19, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], [[1], torch.int32], 2],
            [[np.int8, -1, [32, 64, 1688]], 
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],
                [[6, 7, 31, 9, 10, 11, 12, 19, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]], [[1], torch.int8], 2],
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

instantiate_device_type_tests(TestMaskedFillRange, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()
