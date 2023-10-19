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

import copy
import torch
import numpy as np

from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor


class TestIndexSelect(TestCase):
    def cpu_op_exec(self, input1, axis, indices):
        '''the shape of input:float16, float32,int8,uint8,int32,uint32,int16,uint16,int64,uint64,'''
        output = torch.index_select(input1, dim=axis, index=indices)
        output = output.numpy()
        return output

    def npu_op_exec(self, input1, axis, indices):
        output = torch.index_select(input1, dim=axis, index=indices)
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_out_exec(self, input1, axis, indices, output):
        '''the shape of input:float16, float32,int8,uint8,int32,uint32,int16,uint16,int64,uint64,'''
        torch.index_select(input1, dim=axis, index=indices, out=output)
        output = output.numpy()
        return output

    def npu_op_out_exec(self, input1, axis, indices, output):
        torch.index_select(input1, dim=axis, index=indices, out=output)
        output = output.to('cpu')
        output = output.numpy()
        return output

    def test_index_select(self, device="npu"):
        shape_format = [
            [[np.float32, 0, (3, )], torch.tensor(0, dtype=torch.int64), 0],
            [[np.float32, 0, (3, )], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.float32, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.float32, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.float32, 3, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.float32, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.int8, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.int8, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.int8, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.int8, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.int8, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.uint8, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.uint8, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.uint8, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.uint8, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.uint8, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.int32, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.int32, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.int32, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.int32, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.int32, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.uint8, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.uint8, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.uint8, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.uint8, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.uint8, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.uint8, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.uint8, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.uint8, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.uint8, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.uint8, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],

            [[np.int16, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.int16, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.int16, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.int16, 0, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.int16, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],
        ]
        for item in shape_format:
            input1, npu_input = create_common_tensor(item[0], 1, 100)
            _, npu_out = create_common_tensor(item[0], 1, 100)
            cpu_output = self.cpu_op_exec(input1, item[2], item[1])
            npu_output = self.npu_op_exec(npu_input, item[2], item[1].to('npu'))
            npu_output_out = self.npu_op_out_exec(npu_input, item[2], item[1].to('npu'), npu_out)
            self.assertRtolEqual(cpu_output, npu_output)
            self.assertRtolEqual(cpu_output, npu_output_out)

    def test_index_select_fp16(self, device="npu"):
        shape_format = [
            [[np.float16, 0, (3,)], torch.tensor([0, 1], dtype=torch.int64), 0],
            [[np.float16, 0, (2, 4)], torch.tensor([0, 1, 2], dtype=torch.int64), 1],
            [[np.float16, 0, (3, 4, 6)], torch.tensor([1, 2, 4], dtype=torch.int64), 2],
            [[np.float16, 3, (4, 5, 6, 7)], torch.tensor([3, 5, 6], dtype=torch.int64), 3],
            [[np.float16, -1, (3, 4, 8, 9, 12)], torch.tensor([2, 3, 5, 6], dtype=torch.int64), 4],
            [[np.float16, 0, (3, )], torch.tensor(0, dtype=torch.int64), 0],
        ]
        for item in shape_format:
            input1, npu_input = create_common_tensor(item[0], 1, 100)
            input1 = input1.to(torch.float32)
            cpu_output = self.cpu_op_exec(input1, item[2], item[1])
            npu_output = self.npu_op_exec(npu_input, item[2], item[1].to('npu'))
            cpu_output = cpu_output.astype(np.float16)
            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
